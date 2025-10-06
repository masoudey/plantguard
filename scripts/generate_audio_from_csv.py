"""Generate speech audio files from a CSV of symptom descriptions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
from typing import Iterable, Optional, Tuple

from gtts import gTTS
from gtts.tts import gTTSError
from pydub import AudioSegment


def _change_speed(audio: AudioSegment, speed: float) -> AudioSegment:
    if speed <= 0 or speed == 1.0:
        return audio
    altered = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed)})
    return altered.set_frame_rate(audio.frame_rate)


def _change_pitch(audio: AudioSegment, semitone_shift: float) -> AudioSegment:
    if semitone_shift == 0:
        return audio
    factor = 2.0 ** (semitone_shift / 12.0)
    shifted = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * factor)})
    return shifted.set_frame_rate(audio.frame_rate)


def synthesize(
    text: str,
    destination: Path,
    language: str,
    sample_rate: int,
    mono: bool,
    *,
    tld: str,
    speed: float,
    pitch: float,
    max_retries: int,
    retry_delay: float,
) -> None:
    tmp_mp3 = destination.with_suffix('.mp3')

    attempt = 0
    while True:
        try:
            tts = gTTS(text=text, lang=language, tld=tld)
            tts.save(str(tmp_mp3))
            break
        except (gTTSError, ConnectionError) as exc:  # pragma: no cover - network interaction
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"Failed to synthesize '{destination}' after {max_retries} retries") from exc
            wait = retry_delay * attempt
            print(f"Retrying {destination.name} due to {exc}. Sleeping {wait:.1f}s...")
            time.sleep(wait)

    audio = AudioSegment.from_mp3(tmp_mp3)
    audio = _change_speed(audio, speed)
    audio = _change_pitch(audio, pitch)
    if mono:
        audio = audio.set_channels(1)
    if sample_rate:
        audio = audio.set_frame_rate(sample_rate)

    audio.export(destination, format='wav')
    tmp_mp3.unlink(missing_ok=True)


def iter_rows(
    csv_path: Path,
    text_col: str,
    label_col: str,
    path_col: Optional[str],
    lang_col: Optional[str],
    default_lang: str,
    speed_col: Optional[str],
    pitch_col: Optional[str],
    tld_col: Optional[str],
    default_speed: float,
    default_pitch: float,
    default_tld: str,
) -> Iterable[Tuple[str, str, str, float, float, str]]:
    with csv_path.open(encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        required = {text_col, label_col}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV must contain columns {missing}. Found: {reader.fieldnames}")

        for idx, row in enumerate(reader, 1):
            text = row[text_col].strip()
            label = row[label_col].strip()
            if not text or not label:
                continue

            rel_path = None
            if path_col and path_col in row and row[path_col].strip():
                rel_path = row[path_col].strip().lstrip('/\\')
            if rel_path is None:
                safe_label = label.replace('/', '-').replace(' ', '_')
                rel_path = f"{safe_label}/{safe_label}_{idx:04d}.wav"

            language = default_lang
            if lang_col and lang_col in row and row[lang_col].strip():
                language = row[lang_col].strip()

            speed = default_speed
            if speed_col and speed_col in row and row[speed_col].strip():
                try:
                    speed = float(row[speed_col])
                except ValueError:
                    pass

            pitch = default_pitch
            if pitch_col and pitch_col in row and row[pitch_col].strip():
                try:
                    pitch = float(row[pitch_col])
                except ValueError:
                    pass

            tld_value = default_tld
            if tld_col and tld_col in row and row[tld_col].strip():
                tld_value = row[tld_col].strip()

            yield rel_path, text, language, speed, pitch, tld_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WAV clips for the audio classifier from a CSV file.")
    parser.add_argument("--csv", default="plantguard/data/audio/Expanded_Plant_Symptom_TTS_Manifest.csv", help="Path to the input CSV file.")
    parser.add_argument("--output-dir", default="plantguard/data/processed/audio", help="Base directory to store generated WAV files.")
    parser.add_argument("--text-column", default="text", help="Column name containing the spoken symptom description.")
    parser.add_argument("--label-column", default="label", help="Column name containing the class label.")
    parser.add_argument("--path-column", default="rel_audio_path", help="Column with relative output path (e.g., audio/foo.wav).")
    parser.add_argument("--language", default="en", help="Default language code for gTTS synthesis.")
    parser.add_argument("--language-column", default="lang", help="Optional column containing language codes per row.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate in Hz (default: 16 kHz).")
    parser.add_argument("--mono", action='store_true', help="Force mono channel output.")
    parser.add_argument("--speed", type=float, default=1.0, help="Default playback speed multiplier")
    parser.add_argument("--speed-column", default="speed", help="Optional column for per-row speed multipliers")
    parser.add_argument("--pitch", type=float, default=0.0, help="Default semitone shift applied after synthesis")
    parser.add_argument("--pitch-column", default="pitch", help="Optional column for per-row semitone shifts")
    parser.add_argument("--tld", default="com", help="Default Google TTS top-level domain for accent control (e.g., com, co.uk)")
    parser.add_argument("--tld-column", default="tld", help="Optional column for per-row TLD overrides")
    parser.add_argument("--skip-existing", action="store_true", help="Skip rows whose destination WAV already exists")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for transient TTS failures")
    parser.add_argument("--retry-delay", type=float, default=1.5, help="Base delay (seconds) for retry backoff")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV {csv_path} not found.")

    output_dir = Path(args.output_dir)

    for relative_path, text, language, speed, pitch, tld_value in iter_rows(
        csv_path,
        args.text_column,
        args.label_column,
        args.path_column,
        args.language_column,
        args.language,
        args.speed_column,
        args.pitch_column,
        args.tld_column,
        args.speed,
        args.pitch,
        args.tld,
    ):
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and destination.exists():
            print(f"Skipping existing {destination}")
            continue
        synthesize(
            text,
            destination,
            language or args.language,
            args.sample_rate,
            bool(args.mono),
            tld=tld_value or args.tld,
            speed=speed,
            pitch=pitch,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        print(f"Created {destination}")


if __name__ == "__main__":
    main()
