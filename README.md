# Sound Radar

> ⚠️ **Compliance notice**: This is an audio-analysis learning project. Using
> sound-radar tools in some multiplayer competitive games may violate the
> game's Terms of Service or trigger anti-cheat. **You are responsible for
> verifying the rules of any game you use this with.** See
> [`docs/07_safety_and_compliance.md`](docs/07_safety_and_compliance.md).

A real-time acoustic source direction radar for FPS games.

## What it does

Sound Radar captures the audio output of a game (via WASAPI loopback on Windows), analyzes the multichannel PCM stream, and displays a radar showing the bearing of transient sound events (footsteps, gunshots, etc.) in real time.

## Why multichannel?

When a game is set to output **7.1 / 5.1 surround** (and HRTF is disabled), the horizontal bearing of each in-game sound source is physically encoded as an energy distribution across the discrete speaker channels. This makes direction estimation:

- **Simple**: bearing ≈ argmax(channel energy)
- **Robust**: no fragile HRTF deconvolution
- **Interpretable**: every behavior can be traced to physics, not a black-box model

See `docs/01_solution.md` and `docs/04_audio_basics.md` for the full rationale.

## Quick start

### 1. Prerequisites

- Windows 10/11
- Python 3.13 recommended (3.9+ works)
- A way to obtain a multichannel loopback signal. Recommended:
  - A virtual sound card such as **VoiceMeeter (Banana)** configured as a 7.1 device, **or**
  - Your real audio device configured as 7.1 in Windows Sound Settings.

### 2. Install (use a virtual environment — do NOT use global Python)

The project must be installed in an isolated venv. Local machine has both
Python 3.9 and 3.13; **prefer 3.13**.

    ::powershell
    :: Create the venv with Python 3.13
    py -3.13 -m venv .venv

    :: Activate (PowerShell)
    .\.venv\Scripts\Activate.ps1

    :: Install dependencies inside the venv
    python -m pip install -r requirements.txt

All subsequent commands (`python ...`, `pytest`) must be run with the venv
activated, or by invoking `.\.venv\Scripts\python.exe` directly.

### 3. Configure the game

In the target game's audio settings:

- Speaker configuration: **7.1 Surround** (or 5.1)
- HRTF / spatial audio: **OFF**

### 4. Run

    ::powershell
    python src/main.py

Adjust the **Sensitivity** slider in the UI:
- Drag **right** (higher) = **more sensitive** (picks up quieter / farther sounds)
- Drag **left** (lower) = **less sensitive** (rejects ambient noise)

## Project layout

    sound_radar/
    ├── config.yaml              # default config (game-agnostic)
    ├── docs/                    # design documents + safety/compliance
    ├── src/
    │   ├── audio/               # multichannel loopback capture
    │   ├── analysis/            # features, onset detection, VBAP direction
    │   ├── tracking/            # temporal smoothing
    │   └── ui/                  # PyQt6 radar
    ├── tests/                   # unit tests
    ├── tools/                   # recording / calibration utilities
    └── profiles/                # (optional) per-game calibration results

See `docs/02_architecture.md` for details.

## Design principle: game-agnostic

This software intentionally avoids game-specific tuning:

- Thresholds are **adaptive** (derived from a live noise-floor estimate), not hard-coded dB values.
- Event detection uses **generic acoustic features** (spectral-flux onsets), not game-specific spectral signatures.
- Speaker layout is **fully configurable** (`config.yaml`).
- An optional **calibration tool** (`tools/calibrate.py`) can produce a profile for a specific game, but the core never depends on any profile.

## License

MIT