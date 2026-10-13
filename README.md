# StereogramCLI — AI-Generated Autostereograms

A Python CLI that turns a text prompt into a "Magic Eye" autostereogram.
Generative AI produces a depth map and a seamless texture; a custom NumPy engine encodes the 3D illusion.

## How it works
1. **Prompt → depth map** via Hugging Face / Replicate models (`ai_generators.py`).
2. **Prompt → tiling pattern** for the texture.
3. **Stereo engine** (`stereo_engine.py`) shifts pixels per depth to create the hidden 3D shape.
4. **Batch mode** reads `rally_data.json` to render a full set (see `rally_output/` — a science-themed rally of 15 stereograms for students).

## Usage
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HUGGINGFACE_API_TOKEN=...   # never commit this
python main.py generate --prompt "3D vector"
```

**Stack:** Python · NumPy · Pillow · Click · Hugging Face · Replicate
