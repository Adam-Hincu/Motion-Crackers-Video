"""Composes and mixes the Motion Crackers launch track from real recorded samples.

120 BPM, E dorian. Every hit is placed on the same cue times the video uses
(see video/index.html), so picture and music share one grid.

Sources (all CC0): Virtuosity Drums, Karoryfer Black And Blue Basses,
Karoryfer Black And Green Guitars, Karoryfer Bear Sax, VCSL percussion.

usage: python tools/music.py <samples_dir> <out.wav>
"""
import glob, os, subprocess, sys
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d

SR = 48000
DUR = 18.75
BEAT = 0.5
S16 = BEAT / 4

ROOT = sys.argv[1]
OUT = sys.argv[2]
rand = np.random.default_rng(11)

NAMES = ["c", "db", "d", "eb", "e", "f", "gb", "g", "ab", "a", "bb", "b"]


def nn(midi):
    return f"{NAMES[midi % 12]}{midi // 12 - 1}"


def M(name):
    """'E2' -> 40"""
    name = name.lower()
    base = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[name[0]]
    rest = name[1:]
    if rest.startswith("#"):
        base += 1; rest = rest[1:]
    elif rest.startswith("b") and len(rest) > 1:
        base -= 1; rest = rest[1:]
    return base + (int(rest) + 1) * 12


# ---------------------------------------------------------------- samples
_cache = {}


def load(path):
    if path in _cache:
        return _cache[path]
    try:
        x, sr = sf.read(path, always_2d=True, dtype="float32")
    except sf.LibsndfileError:
        # a few files use a WAVE_FORMAT_EXTENSIBLE variant libsndfile rejects; decode with ffmpeg
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
        x, sr = np.frombuffer(raw, np.float32).reshape(-1, 2).copy(), SR
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    if sr != SR:
        g = np.gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
    # trim to the attack so hits land exactly on the grid
    env = np.abs(x).max(1)
    pk = env.max() + 1e-9
    on = int(np.argmax(env > pk * 0.03))
    on = max(0, on - int(0.002 * SR))
    x = x[on:]
    fade = min(len(x), int(0.0015 * SR))
    x[:fade] *= np.linspace(0, 1, fade)[:, None]
    _cache[path] = x
    return x


def pick(pattern, i=None):
    files = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    if i is None:
        i = int(rand.integers(len(files)))
    # a few takes in the source libraries are corrupt; fall back to a sibling round-robin
    for k in range(len(files)):
        path = files[(i + k) % len(files)]
        if path in BAD:
            continue
        try:
            return load(path)
        except (sf.LibsndfileError, subprocess.CalledProcessError):
            BAD.add(path)
    raise FileNotFoundError(f"no readable take for {pattern}")


BAD = set()


# ---------------------------------------------------------------- buses
def bus():
    return np.zeros((int(DUR * SR) + SR * 3, 2), np.float32)


B = {k: bus() for k in ["kick", "snare", "hats", "perc", "bass", "gtr", "sax", "fx"]}


def place(name, x, t, gain=1.0, pan=0.0, length=None, release=0.04):
    if length is not None:
        n = min(len(x), int(length * SR))
        y = x[:n].copy()
        r = min(n, int(release * SR))
        y[n - r:] *= np.linspace(1, 0, r)[:, None] ** 2
    else:
        y = x
    l = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
    r_ = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
    i = int(round(t * SR))
    if i < 0:
        y = y[-i:]; i = 0
    j = min(len(B[name]), i + len(y))
    B[name][i:j, 0] += y[: j - i, 0] * gain * l
    B[name][i:j, 1] += y[: j - i, 1] * gain * r_


def hum(ms=4):
    return float(rand.normal(0, ms / 1000))


# ---------------------------------------------------------------- instruments
VD = "virtuosity_drums/Samples/"


def kick(t, v=4, g=1.0):
    rr = int(rand.integers(1, 5))
    place("kick", pick(f"{VD}kickmic/kick/kickmic_kick_snoff_vl{v}_rr{rr}.flac"), t, 0.9 * g)
    place("kick", pick(f"{VD}mid/kick/mid_kick_snoff_vl{v}_rr{rr}.flac"), t, 0.9 * g)
    place("kick", pick(f"{VD}room/kick/room_kick_snoff_vl{v}_rr{rr}.flac"), t, 0.45 * g)


def snare(t, v=30, g=1.0, art="center"):
    place("snare", pick(f"{VD}snaremic/snare/snaremic_snare_{art}_vl{v}.flac"), t, 0.55 * g)
    place("snare", pick(f"{VD}mid/snare/mid_snare_{art}_vl{v}.flac"), t, 0.8 * g)
    place("snare", pick(f"{VD}room/snare/room_snare_{art}_vl{v}.flac"), t, 0.5 * g)


def hat(t, v=3, g=1.0, kind="closed", length=None):
    place("hats", pick(f"{VD}mid/hh/mid_hh_{kind}_vl{v}_rr*.flac"), t + hum(3), 0.7 * g, pan=0.22, length=length, release=0.03)


def crash(t, v=3, g=1.0):
    place("fx", pick(f"{VD}mid/crash/mid_crash_crash_vl{v}_rr*.flac"), t, 0.55 * g, pan=-0.25)
    place("fx", pick(f"{VD}room/crash/room_crash_crash_vl{v}_rr*.flac"), t, 0.35 * g, pan=-0.25)


def tom(t, which="htom", v=12, g=1.0):
    place("snare", pick(f"{VD}mid/{which}/mid_{which}_center_vl{v}.flac"), t, 0.8 * g, pan=-0.2 if which == "htom" else 0.15)


VC = "vcsl/Idiophones/Struck Idiophones/"


def clap(t, g=1.0):
    place("perc", pick(VC + "Claps/Clap_rr*.wav"), t - 0.006, 0.55 * g, pan=-0.3)
    place("perc", pick(VC + "Claps/Clap_rr*.wav"), t + 0.004, 0.55 * g, pan=0.3)


def tamb(t, g=1.0, kind="Hit_v2"):
    place("perc", pick(VC + f"Tambourine 1/Tamb1_{kind}_rr1_Mid.wav"), t + hum(3), 0.45 * g, pan=0.4)


def shaker(t, g=1.0):
    f = "Mid_ShakerDouble_Up_rr*.wav" if rand.random() < 0.5 else "Mid_ShakerDouble_Down_rr*.wav"
    place("perc", pick(VC + "Shaker, Small/" + f), t + hum(4), 0.3 * g, pan=-0.4, length=0.12)


def perc(t, path, g=1.0, pan=0.0, length=None):
    place("perc", pick(VC + path), t, g, pan=pan, length=length)


BB = "karoryfer.black-and-blue-basses/Samples/darkblack/"


def bass(t, note, length=0.24, art="reg", dyn="f", g=1.0):
    n = nn(M(note))
    rr = int(rand.integers(1, 5))
    if art == "reg":
        x = pick(f"{BB}reg/darkblack_{n}_{dyn}_rr{rr}.wav")
    elif art == "ghost":
        x = pick(f"{BB}ghost/darkblack_{n}_ghost_{dyn}_rr*.wav")
    else:
        x = pick(f"{BB}stac/darkblack_{n}_stac_rr{rr}.wav")
    place("bass", x, t + hum(2), 0.85 * g, length=length + 0.05, release=0.05)


GG = "karoryfer.black-and-green-guitars/Samples/green/"
EM9 = ["G3", "D4", "Gb4", "B4"]
A13 = ["G3", "Db4", "Gb4", "B4"]


def gtr(t, notes, g=1.0, strum=0.007, ring=None, up=False, length=0.15):
    order = list(reversed(notes)) if up else notes
    for k, note in enumerate(order):
        n = nn(M(note) + 12)  # Karoryfer guitar files are named an octave above sounding pitch
        if ring:
            x = pick(f"{GG}ord/twang_{n}_f_rr*.wav")
            place("gtr", x, t + k * strum, 0.32 * g, pan=0.3, length=ring, release=0.35)
        else:
            x = pick(f"{GG}stac/staccato_{n}_rr*.wav")
            place("gtr", x, t + k * strum + hum(2), 0.34 * g, pan=0.3, length=length, release=0.06)


BS = "karoryfer.bear-sax/Samples/"


def sax(t, note, v=4, g=1.0, length=None, sustain=False):
    # Bear Sax files sound an octave above their names: "E2" here sounds E3
    n = nn(M(note))
    if sustain:
        x = pick(f"{BS}{n}_looped_f.wav")
        place("sax", x, t, 0.9 * g, pan=-0.12, length=length or 1.0, release=0.5)
    else:
        x = pick(f"{BS}{n}_vl{v}_rr*.wav")
        place("sax", x, t + hum(2), 1.0 * g, pan=-0.12, length=length, release=0.05)


# ---------------------------------------------------------------- arrangement
def T(step):
    return step * S16


def main_groove(t0, t1, anchor=0.0, density=1.0):
    """Live funk kit on a 16th grid whose bar starts at `anchor`."""
    for n in range(int(round((t0 - anchor) / S16)), int(round((t1 - anchor) / S16))):
        t, s = anchor + T(n), n % 16
        if s in (0, 7, 10):
            kick(t, 4 if s == 0 else 3)
        if s in (4, 12):
            snare(t, 32 + int(rand.integers(0, 4)))
            tamb(t, 0.8)
        if density > 0.6 and s in (6, 9, 15):
            snare(t + hum(3), 6 + int(rand.integers(0, 5)), 0.75)
        if s == 14 and density > 0.8:
            hat(t, 3, 0.9, "open", length=0.24)
        elif s % 2 == 0:
            hat(t, 4 if s % 4 == 0 else 3, 0.95)
        else:
            hat(t, 2, 0.65)
        shaker(t, 0.8 if s % 2 else 1.0)


def half_time(t0, t1, anchor):
    """Lighter kit for reading moments: kick on 1 and the "and" of 3, snare on 3, 8th hats."""
    for n in range(int(round((t0 - anchor) / S16)), int(round((t1 - anchor) / S16))):
        t, s = anchor + T(n), n % 16
        if s in (0, 10):
            kick(t, 3, 0.85)
        if s == 8:
            snare(t, 26, 0.8)
        if s % 2 == 0:
            hat(t, 3 if s % 4 == 0 else 2, 0.75)
        shaker(t, 0.6)


E_HALF = [(0, "E2", 2, "reg"), (3, "E3", 1, "stac"), (5, "E2", 1, "ghost"), (6, "G2", 1, "stac"), (7, "A2", 1, "stac")]
A_HALF = [(8, "A2", 2, "reg"), (11, "A3", 1, "stac"), (13, "G2", 1, "ghost"), (14, "B2", 1, "stac"), (15, "D3", 1, "stac")]


def bass_line(t0, t1, anchor=0.0):
    for bar in range(int((t0 - anchor) // 2), int(np.ceil((t1 - anchor) / 2))):
        for s, note, ln, art in E_HALF + A_HALF:
            t = anchor + bar * 2 + T(s)
            if t0 - 1e-6 <= t < t1 - 1e-6:
                bass(t, note, ln * S16 * 0.9, art, "p" if art == "ghost" else "f", 0.8 if art == "ghost" else 1.0)


def chord_at(t, anchor=0.0):
    return EM9 if ((t - anchor) % 2) < 1.0 else A13


def gtr_pattern(t0, t1, anchor=0.0):
    for n in range(int(round((t0 - anchor) / S16)), int(round((t1 - anchor) / S16))):
        s, t = n % 16, anchor + T(n)
        if s in (2, 5, 10, 13):
            gtr(t, chord_at(t, anchor), 0.85 if s in (5, 13) else 1.0, up=s in (5, 13))


def stab(t, chord=None, g=1.0, bass_note=None, k=True):
    gtr(t, chord or chord_at(t), g)
    if bass_note:
        bass(t, bass_note, 0.16, "stac")
    if k:
        kick(t, 3)


# ---------------------------------------------------------------- score
# Cue sheet: the same times as the `Q` object in video/index.html.
Q = dict(s3=2.75, land2=3.25, out2=3.8, s4=4.0, land3=4.75, bloom=5.0, s5=5.75, land4=6.25,
         brk=7.0, drop=7.5, mask=9.0, dive=10.5, card=11.0, flip=11.2, head=11.25,
         copy=12.75, drop2=13.0, make=13.75, collapse=14.75, logo=14.75)
L0 = Q["logo"]

# --- 0.0 - 1.5: a dot changes shape on every beat
kick(0.0, 4); bass(0.0, "E2", 0.42, "reg"); sax(0.0, "E1", 4, 0.8); sax(0.0, "E2", 4, 1.0); sax(0.0, "E3", 4, 0.45)
clap(0.0, 0.7)
stab(0.5, EM9, 1.0, "E3"); clap(0.5, 0.8); hat(0.5, 3, 0.9, "open", length=0.3)
stab(1.0, A13, 1.0, "A2"); clap(1.0, 0.8); hat(1.0, 3, 0.9, "open", length=0.3)
snare(1.25, 8, 0.6); snare(1.375, 14, 0.7)
for st in range(0, 12):
    shaker(T(st), 0.7 + 0.3 * (st % 2 == 0))

# --- 1.5 - 2.75: "AI made this video." — one stab per word, the period lands on the 1
stab(1.5, EM9, 1.0, "E2"); clap(1.5, 0.7)
gtr(1.625, EM9, 0.75, up=True); hat(1.625, 2)
gtr(1.75, A13, 0.9); snare(1.75, 30)
gtr(1.875, EM9, 0.8, up=True); hat(1.875, 2)
main_groove(2.0, Q["brk"])
bass_line(2.0, Q["brk"])
tamb(2.0, 1.0)

# --- "From one prompt." — three words, then the orange period lands
stab(Q["s3"], EM9, 1.0, None, k=False); gtr(Q["s3"] + 0.125, EM9, 0.7, up=True); gtr(Q["s3"] + 0.25, A13, 0.95)
gtr(Q["land2"], ["E4", "B4"], 0.75); tamb(Q["land2"], 0.9)
snare(Q["out2"] + 0.075, 12, 0.6)

# --- "We call it a cracker." — a chord per word, a snare for the period, the horn for the bloom
for i in range(5):
    t = Q["s4"] + i * S16
    gtr(t, chord_at(t), 0.8 + 0.06 * i, up=i % 2 == 1)
snare(Q["land3"], 34)
sax(Q["bloom"], "G2", 4, 0.9, length=0.16); sax(Q["bloom"] + S16, "A2", 4, 0.95, length=0.14); sax(Q["bloom"], "E3", 3, 0.5, length=0.3)
tamb(Q["bloom"], 1.1)

# --- "Bite-size motion design." — the line rolls over: a swish out, three stabs in, the flower lands
hat(Q["s5"] - 0.125, 3, 0.8, "open", length=0.2)
gtr(Q["s5"] + 0.125, A13, 0.95); gtr(Q["s5"] + 0.25, EM9, 0.85, up=True); gtr(Q["s5"] + 0.375, A13, 1.0)
clap(Q["land4"], 0.9); snare(Q["land4"], 30, 0.7); bass(Q["land4"], "E3", 0.12, "stac"); sax(Q["land4"], "E3", 4, 0.6, length=0.16)

# --- break: stop. A reverse cymbal pulls into the drop.
rc = np.ascontiguousarray(pick(f"{VD}mid/crash/mid_crash_crash_vl3_rr1.flac")[: int(1.6 * SR)][::-1])
swell = rc[-int(0.5 * SR):].copy()
cut = int(0.03 * SR)  # the last 30 ms are silent so the drop hits into air
swell[-cut:] = 0
swell[-cut - 240:-cut] *= np.linspace(1, 0, 240)[:, None]
place("fx", swell, Q["brk"], 0.55, pan=0.1)
for i, t in enumerate(np.arange(Q["brk"] + 0.25, Q["drop"] - 0.06, S16 / 2)):
    snare(t, 10 + i * 6, 0.55 + 0.12 * i)

# --- the wall: drop, horns lead
D = Q["drop"]
crash(D, 3, 1.0); kick(D, 4, 1.15); bass(D, "E2", 0.24, "reg"); clap(D, 1.0)
snare(D, 36, 0.9); sax(D, "E2", 4, 0.8); sax(D, "E1", 4, 0.6); tamb(D, 1.0)
main_groove(D + S16, Q["dive"], anchor=D)
bass_line(D + 0.25, Q["dive"], anchor=D)
gtr_pattern(D + 0.25, Q["dive"], anchor=D)
for dt, note, ln in [(0.0, "E3", 0.2), (0.25, "E2", 0.1), (0.375, "G2", 0.1), (0.5, "A2", 0.2), (0.75, "B2", 0.1), (1.0, "D3", 0.2), (1.25, "E3", 0.2)]:
    sax(D + dt, note, 4, 1.0, length=ln + 0.06)
# zoom out of the period
Mz = Q["mask"]
crash(Mz, 2, 0.8); sax(Mz, "E1", 4, 0.7); sax(Mz, "E2", 4, 1.0); sax(Mz, "E3", 4, 0.5); clap(Mz, 0.8)
sax(Mz + 1.0, "G2", 3, 0.75, length=0.14); sax(Mz + 1.125, "A2", 3, 0.8, length=0.14); sax(Mz + 1.25, "B2", 4, 0.85, length=0.2)
# tom fill as the camera dives into the letter
for i, (dt, w) in enumerate([(0.0, "htom"), (0.125, "htom"), (0.25, "ltom"), (0.375, "ltom")]):
    tom(Q["dive"] + dt, w, 10 + i, 0.9)
bass(Q["dive"], "E3", 0.1, "stac"); bass(Q["dive"] + 0.25, "D3", 0.1, "stac"); bass(Q["dive"] + 0.375, "B2", 0.1, "stac")

# --- the card: half-time, room to look
Cd = Q["card"]
kick(Cd, 4); crash(Cd, 1, 0.55); bass(Cd, "E2", 0.9, "reg", "mf")
half_time(Cd + S16, Q["copy"] - 0.25, anchor=Cd)
perc(Q["flip"], "Cabasa/Cabasa1_Rub_v2_rr1_Mid.wav", 0.6, pan=0.25)
bass(Cd + 0.75, "G2", 0.2, "stac"); bass(Cd + 1.0, "A2", 0.4, "reg", "mf"); bass(Cd + 1.5, "B2", 0.2, "stac")
for i, note in enumerate(["B3", "D4", "E4", "G4", "A4", "B4"]):
    gtr(Q["head"] + i * S16, [note], 0.95, length=0.2)
snare(Q["copy"] - 0.125, 14, 0.6)

# --- "Copy it." — a band hit, the card drops away under the lines
Cp = Q["copy"]
stab(Cp, EM9, 1.15, "E2"); clap(Cp, 1.0); sax(Cp, "E3", 4, 0.8, length=0.18); snare(Cp, 30, 0.6)
gtr(Cp + 0.125, A13, 0.85, up=True)
perc(Q["drop2"], "Guiro/Guiro_Fast_rr1_Mid.wav", 0.6, pan=-0.2, length=0.45)
tom(Q["drop2"] + 0.25, "ltom", 14, 0.8)
half_time(Cp + S16, Q["collapse"], anchor=Cp)
bass_line(Cp + 0.25, Q["make"], anchor=Cp)

# --- "Make your own." — three words, the lines shuffle, then fold into a dot
Mk = Q["make"]
stab(Mk, A13, 1.1, "A2"); clap(Mk, 0.9)
gtr(Mk + 0.125, A13, 0.9, up=True); bass(Mk + 0.125, "B2", 0.1, "stac")
stab(Mk + 0.25, EM9, 1.1, "D3"); snare(Mk + 0.25, 32)
for i in range(6):  # the lines hop as they are rearranged
    gtr(Mk + 0.25 + 0.0625 + i * 0.0625, [["B4", "A4", "Gb4", "E4", "D4", "B3"][i]], 0.55, length=0.08)
bass(Mk + 0.5, "E2", 0.3, "reg", "mf"); bass(Mk + 0.75, "G2", 0.2, "stac")
perc(Q["collapse"] - 0.25, "Tambourine 1/Tamb1_Roll_v2_rr1_Mid.wav", 0.5, pan=0.3, length=0.3)

# --- the glitch: chopped chord tones on 32nds, getting louder
pool = ["G3", "B3", "D4", "E4", "Gb4", "A4", "B4"]
t, i = L0 + 0.3, 0
while t < L0 + 0.92:
    g = 0.7 + 0.8 * (t - L0 - 0.3) / 0.62
    if rand.random() < 0.82:
        gtr(t, [pool[int(rand.integers(len(pool)))]], g * 0.9, length=0.07)
    if i % 2 == 0:
        hat(t, 2, 0.45 * g)
    if rand.random() < 0.25:
        snare(t, 4 + int(rand.integers(0, 6)), 0.5 * g)
    t += S16 / 2
    i += 1

# --- the logo lands: everything on one hit, then let it ring under the link
F = L0 + 1.0
kick(F, 4, 1.05); crash(F, 3, 0.9); clap(F, 0.9); tamb(F, 1.0)
bass(F, "E2", 2.6, "reg", "f")
sax(F, "E1", 4, 0.75); sax(F, "E2", 4, 0.9); sax(F + 0.02, "E2", 4, 0.75, length=2.1, sustain=True)
gtr(F, ["E3", "B3", "D4", "Gb4", "B4"], 1.0, strum=0.012, ring=2.9)
U = L0 + 2.25  # the link rises in: one soft, slow strum over the ringing chord
gtr(U, ["B3", "D4", "Gb4", "B4"], 0.45, strum=0.03, ring=1.7, up=True); bass(U, "B2", 1.6, "reg", "mf", 0.5)


# ---------------------------------------------------------------- mix
def hp(x, f, order=2):
    sos = signal.butter(order, f, "hp", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def lp(x, f, order=2):
    sos = signal.butter(order, f, "lp", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def peak(x, f, gain_db, q=1.0):
    # RBJ peaking EQ
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f / SR; al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]; a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0)


def shelf_hi(x, f, gain_db):
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f / SR; al = np.sin(w) / 2 * np.sqrt(2)
    c = np.cos(w); sA = 2 * np.sqrt(A) * al
    b = [A * ((A + 1) + (A - 1) * c + sA), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sA)]
    a = [(A + 1) - (A - 1) * c + sA, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sA]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0)


def comp(x, thr_db, ratio, att=0.005, rel=0.08, makeup_db=0.0, key=None):
    k = np.abs(key if key is not None else x).max(1)
    a_ = np.exp(-1 / (att * SR)); r_ = np.exp(-1 / (rel * SR))
    env = np.zeros_like(k)
    e = 0.0
    for i, v in enumerate(k):
        e = a_ * e + (1 - a_) * v if v > e else r_ * e + (1 - r_) * v
        env[i] = e
    lvl = 20 * np.log10(env + 1e-9)
    over = np.maximum(0, lvl - thr_db)
    g = 10 ** ((-over * (1 - 1 / ratio) + makeup_db) / 20)
    return x * g[:, None]


def limit(x, ceil_db, look=0.003, rel=0.06):
    """Look-ahead peak limiter: gain dips smoothly ahead of each peak, never past the ceiling."""
    ceil = 10 ** (ceil_db / 20)
    need = np.minimum(1.0, ceil / np.maximum(np.abs(x).max(1), 1e-9))
    n = int(look * SR)
    g = minimum_filter1d(need, size=2 * n + 1)
    a_, r = np.exp(-1 / (look * SR / 3)), np.exp(-1 / (rel * SR))
    sm = np.empty_like(g)
    e = 1.0
    for i, v in enumerate(g):
        e = a_ * e + (1 - a_) * v if v < e else r * e + (1 - r) * v
        sm[i] = e
    return x * np.minimum(sm, g)[:, None]


def sat(x, drive=1.5):
    return np.tanh(x * drive) / np.tanh(drive)


def room_ir(length=1.1, decay=0.42, seed=3):
    r = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = r.standard_normal((n, 2)) * np.exp(-t / decay * 6.9)[:, None]
    ir = lp(ir, 6500); ir = hp(ir, 180)
    ir[: int(0.012 * SR)] = 0  # pre-delay
    return ir / np.abs(ir).sum(0).max() * 6


IR = room_ir()


def verb(x, wet):
    y = np.stack([signal.fftconvolve(x[:, c], IR[:, c])[: len(x)] for c in range(2)], 1)
    return y * wet


kickb = comp(hp(peak(peak(B["kick"], 60, 3.0, 0.9), 380, -4, 1.2), 30), -14, 3, 0.004, 0.09, 2)
snareb = comp(peak(hp(B["snare"], 90), 200, 2, 1.0), -18, 3, 0.003, 0.1, 2)
hatsb = shelf_hi(hp(B["hats"], 400), 9000, 1.5)
percb = hp(B["perc"], 250)
bassb = comp(lp(hp(peak(B["bass"], 90, 2.5, 0.8), 38), 4500), -18, 4, 0.004, 0.12, 3)
# light pump from the kick on the bass, stabs and horn
duck_key = B["kick"] * 3
bassb = comp(bassb, -20, 3, 0.002, 0.11, 0, key=duck_key)
gtrb = lp(peak(hp(B["gtr"], 160), 2600, 2.0, 0.9), 7500, 4)  # keep the bite, drop the hiss above it
gtrb = comp(gtrb, -20, 3, 0.003, 0.08, 2)
saxb = comp(peak(hp(sat(B["sax"], 1.8), 55), 900, 2, 0.8), -16, 3, 0.006, 0.12, 2)
fxb = hp(B["fx"], 300)

def active_rms_db(x):
    m = np.abs(x).max(1)
    win = int(0.05 * SR)
    env = np.sqrt(np.convolve(m ** 2, np.ones(win) / win, "same"))
    act = env > env.max() * 10 ** (-30 / 20)
    return 20 * np.log10(np.sqrt((m[act] ** 2).mean()) + 1e-12)


def level(x, target_db):
    return x * 10 ** ((target_db - active_rms_db(x)) / 20)


# relative mix (dB, measured while each part plays)
kickb = level(kickb, -15.0)
snareb = level(snareb, -19.0)
hatsb = level(hatsb, -29.0)
percb = level(percb, -27.0)
fxb = level(fxb, -27.0)
bassb = level(bassb, -17.0)
gtrb = limit(level(gtrb, -22.5), -11.0)
saxb = level(saxb, -18.5)

drums = kickb + snareb + hatsb + percb + fxb
drums = comp(sat(drums, 1.3), -12, 2.5, 0.003, 0.08, 1.5)
music = bassb + gtrb + saxb
mix = drums + music
mix += verb(snareb * 0.6 + gtrb * 0.5 + saxb * 0.6 + percb * 0.4, 0.28)

# master: glue, gentle tilt, soft clip, loudness
mix = comp(mix, -10, 2.0, 0.01, 0.15, 1.0)
mix = shelf_hi(mix, 11000, 1.0)

# section rides: verse sits under the drop, the card section breathes so the text can be read
ride_db = np.zeros(len(mix))
for t0, t1, db in [(1.98, Q["brk"] - 0.02, -1.0), (Q["card"] - 0.02, Q["copy"] - 0.05, -2.0)]:
    a_, b_ = int(t0 * SR), int(t1 * SR)
    ride_db[a_:b_] = db
k = int(0.02 * SR)
ride_db = np.convolve(ride_db, np.ones(k) / k, "same")
mix *= (10 ** (ride_db / 20))[:, None]
mix = mix[: int(DUR * SR)]
# fade the very end so the ring-out closes cleanly with the picture
fo = int(0.7 * SR)
mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5

try:
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    lufs = meter.integrated_loudness(mix)
    mix *= 10 ** ((-14.0 - lufs) / 20)
except Exception:
    mix /= np.abs(mix).max() / 0.7
pre = np.abs(mix).max(1)
mix = limit(mix, -1.5)  # clean look-ahead limiting, no clipping, true-peak headroom for AAC
gr = 20 * np.log10(np.maximum(np.abs(mix).max(1), 1e-9) / np.maximum(pre, 1e-9))
print("master limiter: max gain reduction %.1f dB" % -gr[pre > 1e-3].min())
sf.write(OUT, mix.astype(np.float32), SR, subtype="PCM_24")
print("wrote", OUT, "peak", round(20 * np.log10(np.abs(mix).max()), 2), "dBFS")
