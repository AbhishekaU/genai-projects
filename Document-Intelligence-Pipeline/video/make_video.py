"""
Builds the final video from the rendered slides and narration audio:

    python video/make_video.py            # voice + render + assemble
    python video/make_video.py --no-render  # reuse existing slide PNGs

Output: video/build/document-intelligence-pipeline.mp4 (1920x1080, 30fps,
H.264 + AAC -- what LinkedIn expects).

Per slide: a short lead-in of silence, the narration, then a short tail; the
slide fades in and out. Subtitles are drawn into the frames (most LinkedIn
viewers scroll with sound off), one sentence-sized chunk at a time, each shown
for a share of the narration proportional to its length.
"""

import math
import re
import struct
import subprocess
import sys
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from narration import SLIDES  # noqa: E402

BUILD = Path(__file__).parent / "build"
AUDIO_DIR, SLIDE_DIR = BUILD / "audio", BUILD / "slides"
WORK = BUILD / "work"
OUTPUT = BUILD / "document-intelligence-pipeline.mp4"

LEAD_IN, TAIL = 0.4, 0.7      # seconds of silence around each slide's narration
FADE = 0.3                    # seconds of fade in / out per slide
FPS = 30
SAMPLE_RATE = 24_000

FONT = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 40)  # Segoe UI Semibold
CAPTION_MAX_W = 1500
CAPTION_CENTER_Y = 975


def split_caption(text: str, max_words: int = 14) -> list[str]:
    """Sentence-sized chunks, with long sentences split at a comma or colon."""
    chunks = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sentence.split()
        while len(words) > max_words:
            # Break after the last comma/colon inside the first max_words, if any.
            cut = max((i for i, w in enumerate(words[:max_words]) if w.endswith((",", ":"))), default=max_words - 1)
            chunks.append(" ".join(words[:cut + 1]))
            words = words[cut + 1:]
        if words:
            chunks.append(" ".join(words))
    return chunks


def wrap(draw: ImageDraw.ImageDraw, text: str) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=FONT) <= CAPTION_MAX_W:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def caption_frame(slide_png: Path, text: str, out: Path) -> None:
    img = Image.open(slide_png).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    lines = wrap(draw, text)
    line_h = 54
    width = max(draw.textlength(l, font=FONT) for l in lines)
    height = line_h * len(lines)
    x0, y0 = (1920 - width) / 2 - 32, CAPTION_CENTER_Y - height / 2 - 18
    draw.rounded_rectangle((x0, y0, x0 + width + 64, y0 + height + 36), radius=16,
                           fill=(26, 26, 25, 235), outline=(255, 255, 255, 30))
    for i, l in enumerate(lines):
        w = draw.textlength(l, font=FONT)
        draw.text(((1920 - w) / 2, y0 + 14 + i * line_h), l, font=FONT, fill=(240, 239, 236))
    img.save(out)


def trim_tail_artifact(pcm: bytes, rate: int) -> bytes:
    """Gemini TTS ends every clip with ~130ms of loud noise *after* the speech
    has gone silent -- audible as a scratch at the end of each slide. Find the
    last silent gap (>= 80ms) in the final half second and cut just after it
    starts, with a 20ms fade so the cut can't click. A clip that doesn't end
    that way is returned unchanged."""
    samples = list(struct.unpack(f"<{len(pcm) // 2}h", pcm))
    win = rate // 100  # 10ms windows

    def quiet(i: int) -> bool:
        chunk = samples[i:i + win]
        rms = math.sqrt(sum(x * x for x in chunk) / len(chunk)) / 32768
        return rms < 10 ** (-45 / 20)  # below -45 dBFS

    start = max(0, len(samples) - rate // 2)
    windows = list(range(start, len(samples) - win, win))
    gap_start, tail_start, run = None, None, 0
    for i in windows:
        if quiet(i):
            run += 1
            if run == 8 and gap_start is None:      # 80ms of silence
                gap_start = i - 7 * win
        elif gap_start is not None:
            tail_start = i                          # the loud tail after the gap
            break
        else:
            run = 0
    # Only a short burst running to the very end is the artifact. Anything
    # longer could be a last word after a pause -- leave that alone.
    if gap_start is None or tail_start is None or len(samples) - tail_start > rate // 5:
        return pcm

    cut = gap_start + 6 * win  # keep 60ms of the silence
    fade = rate // 50          # 20ms
    out = samples[:cut]
    for k in range(fade):
        out[-fade + k] = int(out[-fade + k] * (1 - k / fade))
    return struct.pack(f"<{len(out)}h", *out)


def padded_audio(src: Path, out: Path) -> float:
    """Trims the TTS tail artifact, adds the lead-in and tail silence; returns
    the narration's own length."""
    with wave.open(str(src)) as f:
        pcm, rate = f.readframes(f.getnframes()), f.getframerate()
    pcm = trim_tail_artifact(pcm, rate)
    silence = lambda s: b"\x00\x00" * int(rate * s)  # noqa: E731 -- 16-bit mono
    with wave.open(str(out), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(silence(LEAD_IN) + pcm + silence(TAIL))
    return len(pcm) / 2 / rate


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def build_segment(index: int) -> Path:
    slide = SLIDES[index]
    seg_dir = WORK / slide.id
    seg_dir.mkdir(parents=True, exist_ok=True)

    speech = padded_audio(AUDIO_DIR / f"{slide.id}.wav", seg_dir / "audio.wav")
    total = LEAD_IN + speech + TAIL

    # Time each caption chunk by its share of the characters spoken.
    chunks = split_caption(slide.subtitle)
    chars = sum(len(c) for c in chunks)
    timeline, t = [], 0.0
    for i, chunk in enumerate(chunks):
        frame = seg_dir / f"cap{i}.png"
        caption_frame(SLIDE_DIR / f"{slide.id}.png", chunk, frame)
        share = speech * len(chunk) / chars
        # The first chunk also covers the lead-in, the last one the tail.
        duration = share + (LEAD_IN if i == 0 else 0) + (TAIL if i == len(chunks) - 1 else 0)
        timeline.append((frame, duration))
        t += duration

    concat = seg_dir / "frames.txt"
    lines = [f"file '{f.as_posix()}'\nduration {d:.3f}" for f, d in timeline]
    lines.append(f"file '{timeline[-1][0].as_posix()}'")  # concat demuxer needs the last frame repeated
    concat.write_text("\n".join(lines), encoding="utf-8")

    out = seg_dir / "segment.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-i", str(seg_dir / "audio.wav"),
         "-vf", f"fps={FPS},format=yuv420p,fade=t=in:st=0:d={FADE},fade=t=out:st={total - FADE:.3f}:d={FADE}",
         "-af", f"afade=t=out:st={total - FADE:.3f}:d={FADE}",
         "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(out)])
    print(f"built {slide.id}  {total:4.1f}s  {len(chunks)} captions")
    return out


def main() -> None:
    if "--no-render" not in sys.argv:
        import render_slides
        import voice
        voice.generate_all()
        render_slides.render_all()

    segments = [build_segment(i) for i in range(len(SLIDES))]
    listing = WORK / "segments.txt"
    listing.write_text("\n".join(f"file '{s.as_posix()}'" for s in segments), encoding="utf-8")

    # Join, and normalize loudness to -14 LUFS (where LinkedIn/most platforms sit).
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
         "-movflags", "+faststart", str(OUTPUT)])
    print(f"\nvideo -> {OUTPUT}")


if __name__ == "__main__":
    main()
