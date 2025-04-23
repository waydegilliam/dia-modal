from __future__ import annotations

import pathlib
import tempfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import sounddevice as sd
import soundfile as sf
import typer
from pydub import AudioSegment

DATA_DIR = pathlib.Path(__file__).parent / "data" / "audio" / "recordings"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 44_100
CHANNELS = 1
BLOCKSIZE = 1024
DTYPE = "int16"

app = typer.Typer(add_completion=False)


def _timestamped_mp3_path() -> pathlib.Path:
    """Return DATA_DIR / output_MM-DD-YYYY_HH-MM-SS.mp3 (PST/PDT)."""
    now_la = datetime.now(ZoneInfo("America/Los_Angeles"))
    ts = now_la.strftime("%m-%d-%Y_%H-%M-%S")
    return DATA_DIR / f"output_{ts}.mp3"


def _stream_to_wav(wav_path: pathlib.Path) -> None:
    """Capture mic input into *wav_path* until Ctrl‑C."""
    typer.echo("🎙️  Recording…  (press Ctrl‑C to stop)\n")

    with sf.SoundFile(
        wav_path,
        mode="w",
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        subtype="PCM_16",
    ) as wav_file:

        def callback(indata, frames, time_info, status):
            if status:
                typer.echo(str(status), err=True)
            wav_file.write(indata)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=BLOCKSIZE,
            dtype=DTYPE,
            callback=callback,
        ):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                typer.echo("\n🛑  Recording stopped.")


def _wav_to_mp3(wav_path: pathlib.Path, mp3_path: pathlib.Path) -> None:
    typer.echo("🔄  Converting to MP3 …")
    AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3")
    typer.echo(f"💾  Saved {mp3_path.resolve()}")


@app.command(help="Record mic input until Ctrl‑C and save as timestamped MP3.")
def record() -> None:
    output_mp3 = _timestamped_mp3_path()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = pathlib.Path(tmp.name)

    try:
        _stream_to_wav(tmp_wav)
        _wav_to_mp3(tmp_wav, output_mp3)
    finally:
        tmp_wav.unlink(missing_ok=True)


if __name__ == "__main__":
    app()
