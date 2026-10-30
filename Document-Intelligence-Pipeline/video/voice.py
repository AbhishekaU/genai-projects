"""
Generates the narration audio, one WAV per slide, with Gemini text-to-speech.

    python video/voice.py            # voice any slide whose narration changed
    python video/voice.py --check    # transcribe every clip and compare to the script

Audio is cached in video/build/audio/<slide id>.wav together with the text it
was made from; a slide is only re-voiced when its narration (or the voice)
changes, so tweaking one line doesn't re-generate the other eleven.
"""

import os
import sys
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
from narration import SLIDES  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")

TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
# Charon: Gemini's "informative" male voice -- calm, steady and confident on its
# own. No delivery direction is sent: this model reads any direction placed
# ahead of the text out loud ("Read this calmly...") and rejects a system
# instruction, which is what --check below was added to catch.
VOICE = os.getenv("GEMINI_TTS_VOICE", "Charon")
CHECK_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

AUDIO_DIR = Path(__file__).parent / "build" / "audio"
SAMPLE_RATE = 24_000  # Gemini TTS returns 24 kHz, 16-bit mono PCM


def synthesize(client: genai.Client, text: str) -> bytes:
    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
                )
            ),
        ),
    )
    return response.candidates[0].content.parts[0].inline_data.data


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)


def generate_all() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client()

    for slide in SLIDES:
        wav, stamp = AUDIO_DIR / f"{slide.id}.wav", AUDIO_DIR / f"{slide.id}.txt"
        key = f"{TTS_MODEL}|{VOICE}|{slide.narration}"
        if wav.exists() and stamp.exists() and stamp.read_text(encoding="utf-8") == key:
            continue

        print(f"voicing {slide.id} ...", flush=True)
        for attempt in range(3):
            try:
                pcm = synthesize(client, slide.narration)
                break
            except Exception as e:  # preview TTS models fail transiently now and then
                if attempt == 2:
                    raise
                print(f"    retrying after error: {e!r}"[:200])
                time.sleep(5 * (attempt + 1))
        write_wav(wav, pcm)
        stamp.write_text(key, encoding="utf-8")


def check_all() -> None:
    """Transcribes each clip back with Gemini and compares it to the script, to
    catch the voice saying something other than the narration."""
    from rapidfuzz import fuzz

    client = genai.Client()
    for slide in SLIDES:
        audio = (AUDIO_DIR / f"{slide.id}.wav").read_bytes()
        heard = client.models.generate_content(
            model=CHECK_MODEL,
            contents=[types.Part.from_bytes(data=audio, mime_type="audio/wav"),
                      "Transcribe this audio verbatim. Output only the transcript."],
        ).text.strip()
        # Compared with the subtitle text: the transcriber writes "CUAD" and
        # "46%" the way the captions do, not the way the narration spells them.
        score = fuzz.token_sort_ratio(heard.lower(), slide.subtitle.lower())
        seconds = (len(audio) - 44) / 2 / SAMPLE_RATE
        flag = "ok   " if score >= 85 else "CHECK"
        print(f"{flag} {slide.id:16} {seconds:5.1f}s  match {score:3.0f}  heard: {heard[:90]}")


if __name__ == "__main__":
    check_all() if "--check" in sys.argv else generate_all()
