# Dia on Modal

This repo is a basic module for running [nari-labs/Dia-1.6B](https://github.com/nari-labs/dia?tab=readme-ov-file) on [modal.com](https://modal.com/).

We take the existing hyperparameters and Gradio app and create a modal deployment from this.

## Local Setup

**Prerequisites**
- Macos or Linux
- (ffmpeg)[https://ffmpeg.org/]
- (uv)[https://github.com/astral-sh/uv]
- a (Modal)[https://modal.com/] account

**Installation**

```bash
uv sync --all-extras --all-groups
```

## Running 

### Local

This will automatically use Cuda or MPS if it's available on your system. 

```bash
modal run main.py --input-text "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices."
```

### Remote

1. Deploy to Modal

```bash
modal deploy main.py::modal_app
```

2. Call deployed function

This will run the inference function and download the generated data to `data/audio/generations`.

```bash
uv run main.py "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices."
```