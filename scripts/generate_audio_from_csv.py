"""Generate speech audio files from a CSV of symptom descriptions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional, Tuple

from gtts import gTTS
from pydub import AudioSegment


def synthesize(text: str, destination: Path, language: str, sample_rate: int, mono: bool) -> None:
    tmp_mp3 = destination.with_suffix('.mp3')
    tts = gTTS(text=text, lang=language)
    tts.save(str(tmp_mp3))

    audio = AudioSegment.from_mp3(tmp_mp3)
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
) -> Iterable[Tuple[str, str, str]]:
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

            yield rel_path, text, language


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WAV clips for the audio classifier from a CSV file.")
    parser.add_argument("--csv", default="plantguard/data/audio/Expanded_Plant_Symptom_TTS_Manifest.csv", help="Path to the input CSV file.")
    parser.add_argument("--output-dir", default="plantguard/data/processed/audio/train", help="Directory to store generated WAV files grouped by label.")
    parser.add_argument("--text-column", default="text", help="Column name containing the spoken symptom description.")
    parser.add_argument("--label-column", default="label", help="Column name containing the class label.")
    parser.add_argument("--path-column", default="rel_audio_path", help="Column with relative output path (e.g., audio/foo.wav).")
    parser.add_argument("--language", default="en", help="Default language code for gTTS synthesis.")
    parser.add_argument("--language-column", default="lang", help="Optional column containing language codes per row.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate in Hz (default: 16 kHz).")
    parser.add_argument("--mono", action='store_true', help="Force mono channel output.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV {csv_path} not found.")

    output_dir = Path(args.output_dir)

    for relative_path, text, language in iter_rows(
        csv_path,
        args.text_column,
        args.label_column,
        args.path_column,
        args.language_column,
        args.language,
    ):
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        synthesize(text, destination, language or args.language, args.sample_rate, bool(args.mono))
        print(f"Created {destination}")


if __name__ == "__main__":
    main()
