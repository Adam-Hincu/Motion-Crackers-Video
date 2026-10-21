"""Shape text with HarfBuzz using Switzer at a given weight and emit SVG outlines.

Writes video/assets/mask-gallery.svg (the "A whole gallery of them." mask) and
video/assets/mask-gallery.json (anchor points used by the zoom moves).
"""
import json, sys, io
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen
import uharfbuzz as hb

ROOT = "/home/user/Motion-Crackers-Video/video/assets/"
W, H = 1920, 1080

def static_font(weight):
    f = TTFont(ROOT + "fonts/Switzer-Variable.woff2")
    inst = instantiateVariableFont(f, {"wght": weight})
    inst.flavor = None
    buf = io.BytesIO(); inst.save(buf)
    return TTFont(io.BytesIO(buf.getvalue())), buf.getvalue()

def shape(data, text, size, tracking_em=0.0):
    face = hb.Face(data); font = hb.Font(face)
    upem = face.upem
    buf = hb.Buffer(); buf.add_str(text); buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": True})
    s = size / upem
    out, x = [], 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        out.append((info.codepoint, x + pos.x_offset * s, info.cluster))
        x += pos.x_advance * s + tracking_em * size
    return out, x - tracking_em * size, s

def line_paths(tt, glyphs, s, ox, baseline):
    gs = tt.getGlyphSet(); order = tt.getGlyphOrder()
    parts, boxes = [], []
    for gid, x, cl in glyphs:
        name = order[gid]
        pen = SVGPathPen(gs)
        tp = TransformPen(pen, (s, 0, 0, -s, ox + x, baseline))
        gs[name].draw(tp)
        bp = BoundsPen(gs); gs[name].draw(TransformPen(bp, (s, 0, 0, -s, ox + x, baseline)))
        parts.append(pen.getCommands()); boxes.append((cl, bp.bounds))
    return parts, boxes

def main():
    weight, size, lh, tracking = 800, 280, 0.97, -0.02
    tt, data = static_font(weight)
    lines = ["A whole", "gallery", "of them."]
    asc = 0.72 * size  # approx cap height for vertical centering
    total = lh * size * (len(lines) - 1) + asc
    top_baseline = (H - total) / 2 + asc
    paths, anchors = [], {}
    for i, text in enumerate(lines):
        glyphs, width, s = shape(data, text, size, tracking)
        ox = (W - width) / 2
        base = top_baseline + i * lh * size
        p, boxes = line_paths(tt, glyphs, s, ox, base)
        paths += p
        for cl, b in boxes:
            if b is None: continue
            ch = text[cl]
            key = f"{i}:{cl}:{ch}"
            anchors[key] = [round(v, 2) for v in b]
    d = " ".join(paths)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"><path d="{d}" fill="#000"/></svg>'
    open(ROOT + "mask-gallery.svg", "w").write(svg)
    json.dump(anchors, open(ROOT + "mask-gallery.json", "w"), indent=1)
    for k, v in anchors.items():
        if k.endswith(":.") or k.endswith(":l"): print(k, v)

main()
