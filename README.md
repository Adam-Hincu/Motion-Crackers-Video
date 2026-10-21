# Motion Crackers — launch film

16.5 s, 1920×1080, 30 fps. Final cut: `renders/motion-crackers-launch.mp4`.

- `video/` — HyperFrames composition (`index.html`, one GSAP timeline).
- `tools/text_mask.py` — shapes "A whole gallery of them." in Switzer with HarfBuzz and writes the vector mask the gallery is seen through.
- `tools/music.py` — composes and mixes the score from recorded samples onto the same cue times as the picture.

## Rebuild

```bash
npm install
python tools/text_mask.py
python tools/music.py <samples_dir> video/assets/music.wav
cd video && npx hyperframes check && npx hyperframes render --quality delivery --output ../renders/motion-crackers-launch.mp4
```

`<samples_dir>` holds sparse clones of the sample libraries below (Python deps: numpy, scipy, soundfile, pyloudnorm, fonttools, uharfbuzz, brotli).

## Credits

Type: Switzer (Indian Type Foundry, FFL), Instrument Serif and Geist Mono (OFL).
Recorded instruments, all CC0: Virtuosity Drums (Versilian), Black And Blue Basses, Black And Green Guitars and Bear Sax (Karoryfer Lecolds), VCSL percussion (Versilian Community Sample Library).
