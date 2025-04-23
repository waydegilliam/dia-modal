import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import huggingface_hub
import modal
import modal.gpu
import numpy as np
import soundfile as sf
import structlog
from pydantic import BaseModel, Field
from pydub import AudioSegment

logger = structlog.get_logger()

DATA_DIR = Path(__file__).parent / "data" if modal.is_local() else Path("/data")

MODAL_GPU = "H100"

INPUT_SAMPLE_RATE = 44100
OUTPUT_SAMPLE_RATE = 44100


class ModelHyperparameters(BaseModel):
    max_new_tokens: int = Field(4072, ge=860, le=4072, description="Upper bound on generated audio length")
    cfg_scale: float = Field(3.0, ge=1.0, le=5.0, description="higher = closer to the prompt.")
    temperature: float = Field(1.3, ge=1.0, le=1.5, description="Sampling temperature; lower = more deterministic.")
    top_p: float = Field(0.95, ge=0.80, le=1.0, description="Nucleus‑sampling probability mass.")
    cfg_filter_top_k: int = Field(30, ge=15, le=50, description="Top‑k filter applied during CFG guidance.")
    speed_factor: float = Field(0.94, ge=0.8, le=1.0, description="Playback‑speed. 1.0 = original.")


modal_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .run_commands(["apt update", "apt upgrade -y"])
    .apt_install(["build-essential", "git"])
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["tts"], gpu=MODAL_GPU)
)

modal_volume = modal.Volume.from_name("tts-data", create_if_missing=True)

modal_app = modal.App(name="tts", image=modal_image, volumes={"/data": modal_volume})


def download_model() -> tuple[str, str]:
    data_model_dir = DATA_DIR / "models" / "nari-labs" / "dia-1.6B"

    checkpoint_filepath = huggingface_hub.hf_hub_download(
        repo_id="nari-labs/Dia-1.6B",
        filename="dia-v0_1.pth",
        local_dir=data_model_dir,
    )
    config_filepath = huggingface_hub.hf_hub_download(
        repo_id="nari-labs/Dia-1.6B",
        filename="config.json",
        local_dir=data_model_dir,
    )

    if not modal.is_local():
        modal_volume.commit()

    return checkpoint_filepath, config_filepath


def read_input_audio(path: Path) -> np.typing.NDArray[np.float32]:
    seg: AudioSegment = AudioSegment.from_file(path, format="mp3")

    if seg.frame_rate != INPUT_SAMPLE_RATE:
        seg = seg.set_frame_rate(INPUT_SAMPLE_RATE)

    if seg.channels != 1:
        seg = seg.set_channels(1)

    # Raw PCM to NumPy
    dtype = np.int16 if seg.sample_width == 2 else np.int32
    pcm: np.typing.NDArray[np.int16 | np.int32] = np.frombuffer(seg._data, dtype=dtype)
    pcm = pcm.reshape(-1)  # mono: (n_samples,)

    max_val: float = float(1 << (8 * seg.sample_width - 1))
    audio: np.typing.NDArray[np.float32] = pcm.astype(np.float32) / max_val

    return audio


def process_input_audio(audio_data: np.typing.NDArray[np.float32]) -> Path:
    # Check if audio_data is valid and not empty
    if audio_data.size == 0 or audio_data.max() == 0:
        raise ValueError("Audio prompt seems empty or silent, ignoring prompt.")

    # Save prompt audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
        temp_audio_prompt_path = f_audio.name  # Store path for cleanup

        # Basic audio preprocessing for consistency
        # Convert to float32 in [-1, 1] range if integer type
        if np.issubdtype(audio_data.dtype, np.integer):
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(np.float32) / max_val
        elif not np.issubdtype(audio_data.dtype, np.floating):
            logger.warning("Unsupported audio prompt dtype, attempting conversion.", dtype=audio_data.dtype)
            try:
                audio_data = audio_data.astype(np.float32)
            except Exception as e:
                raise ValueError(f"Failed to convert audio prompt to float32: {e}") from e

        # Ensure mono (average channels if stereo)
        if audio_data.ndim > 1:
            if audio_data.shape[0] == 2:  # Assume (2, N)
                audio_data = np.mean(audio_data, axis=0)
            elif audio_data.shape[1] == 2:  # Assume (N, 2)
                audio_data = np.mean(audio_data, axis=1)
            else:
                logger.warning("Audio prompt has unexpected shape, taking first channel/axis.", shape=audio_data.shape)
                audio_data = audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]
            audio_data = np.ascontiguousarray(audio_data)  # Ensure contiguous after slicing/mean

        # Write using soundfile (Explicitly use FLOAT subtype)
        try:
            sf.write(temp_audio_prompt_path, audio_data, INPUT_SAMPLE_RATE, subtype="FLOAT")
        except Exception as e:
            raise ValueError("Error writing temporary audio file") from e

    processed_input_audio_filepath = Path(temp_audio_prompt_path)
    if not processed_input_audio_filepath.exists():
        raise ValueError("Failed to create temporary audio prompt file")

    return processed_input_audio_filepath


def process_generated_audio(*, audio: np.typing.NDArray[np.float32], hyperparameters: ModelHyperparameters):
    # Slow down audio
    original_len = len(audio)

    # Ensure speed_factor is positive and not excessively small/large to avoid issues
    speed_factor = max(0.1, min(hyperparameters.speed_factor, 5.0))

    # Target length based on speed_factor
    target_len = int(original_len / speed_factor)

    # Only interpolate if length changes and is valid
    output_audio = audio
    if target_len != original_len and target_len > 0:
        x_original = np.arange(original_len)
        x_resampled = np.linspace(0, original_len - 1, target_len)
        resampled_audio_np = np.interp(x_resampled, x_original, audio)
        output_audio = resampled_audio_np.astype(np.float32)
        logger.info("Resampled audio", original_len=original_len, target_len=target_len, speed_factor=speed_factor)

    return output_audio


def save_audio(*, audio) -> Path:
    datetime_now = datetime.now(ZoneInfo("America/Los_Angeles"))
    timestamp_now = datetime_now.strftime("%m-%d-%Y_%H-%M-%S")
    output_filename = f"output_{timestamp_now}.wav"
    output_filepath = DATA_DIR / "audio" / "generations" / output_filename
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    if not modal.is_local():
        sf.write(str(output_filepath), audio, OUTPUT_SAMPLE_RATE)
        modal_volume.commit()
    else:
        sf.write(str(output_filepath), audio, OUTPUT_SAMPLE_RATE)

    logger.info("Saved audio", filepath=output_filepath)

    return output_filepath


@modal_app.cls(gpu=MODAL_GPU, scaledown_window=60 * 5)
class InferenceTTS:
    @modal.enter()
    def load_model(self):
        import torch
        from dia.model import Dia

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        checkpoint_filepath, config_filepath = download_model()

        self.model = Dia.from_local(config_path=config_filepath, checkpoint_path=checkpoint_filepath, device=device)

    @modal.method()
    def generate(
        self,
        *,
        input_text: str,
        input_audio_filepath: Path | None = None,
        hyperparameters: ModelHyperparameters | None = None,
    ):
        hyperparameters = ModelHyperparameters.model_validate({}) if hyperparameters is None else hyperparameters

        logger.info("Generating audio", text=input_text)

        # Generate audio
        output_audio_np = self.model.generate(
            text=input_text,
            max_tokens=hyperparameters.max_new_tokens,
            cfg_scale=hyperparameters.cfg_scale,
            temperature=hyperparameters.temperature,
            top_p=hyperparameters.top_p,
            use_cfg_filter=True,
            cfg_filter_top_k=hyperparameters.cfg_filter_top_k,
            use_torch_compile=False,
            audio_prompt_path=str(input_audio_filepath) if input_audio_filepath else None,
        )
        logger.info("Generated audio", type=type(output_audio_np))

        # Postprocess audio
        if output_audio_np is not None:
            output_audio = process_generated_audio(audio=output_audio_np, hyperparameters=hyperparameters)
            logger.info("Audio conversion successful", type=type(output_audio), final_shape=output_audio.shape)
        else:
            logger.warning("Generation finished, but no valid tokens were produced.")

        # Save processed audio
        output_filepath = save_audio(audio=output_audio)

        return output_filepath


@modal_app.local_entrypoint()
def local_entrypoint(input_text: str, input_audio_filepath: Path | None = None):
    InferenceTTS().generate.local(input_text=input_text, input_audio_filepath=input_audio_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio locally using the InferenceTTS model.")
    parser.add_argument("input_text", type=str, help="Text to synthesize.")
    parser.add_argument(
        "--input-audio-filepath", type=Path, default=None, help="Optional path to an audio file to use as a prompt."
    )

    args = parser.parse_args()

    input_text = args.input_text
    input_audio_filepath = args.input_audio_filepath

    if input_audio_filepath:
        if not input_audio_filepath.exists():
            raise ValueError(f"Input audio file does not exist: {input_audio_filepath}")
        logger.info("Using audio prompt", filepath=input_audio_filepath)

    logger.info("Generating audio", text=input_text)
    cls = modal.Cls.from_name("tts", "InferenceTTS")
    obj = cls()
    output_filepath_str = obj.generate.remote(input_text=input_text, input_audio_filepath=input_audio_filepath)
    output_filepath = Path(output_filepath_str)

    logger.info("Saving audio")
    local_filename = output_filepath.name
    local_output_path = DATA_DIR / "audio" / "generations" / local_filename
    local_output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure local directory exists

    try:
        with open(local_output_path, "wb") as f:
            for chunk in modal_volume.read_file(str(output_filepath.relative_to("/data"))):
                f.write(chunk)
        logger.info("Successfully saved audio locally", filepath=local_output_path)
    except Exception as e:
        logger.error(
            "Failed to read from volume or write locally",
            error=e,
            volume_path=output_filepath,
            local_path=local_output_path,
        )
