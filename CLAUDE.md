# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**SynthioLabs Voice Slide Deck — Backend** — FastAPI backend for a voice-activated interactive slide deck prototype. Handles real-time voice I/O via Deepgram STT and ElevenLabs TTS, LangGraph for agentic state management, and OpenAI for LLM.

**Stack:** Python 3.11+, FastAPI, LangGraph, Deepgram SDK, ElevenLabs SDK, OpenAI SDK

## Commands

```bash
# Virtual environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Testing
pytest                           # Run all tests
pytest -v tests/                 # Verbose
pytest --cov=src --cov-report=html  # With coverage
pytest -k "test_function_name"   # Single test

# Linting and formatting
ruff check .                     # Lint
ruff format .                    # Format
mypy src/                        # Type checking

# Run server
uvicorn app.main:app --reload --port 8000
```

## Architecture

```
app/
  main.py               # FastAPI app entrypoint
  api/
    routes.py           # HTTP + WebSocket endpoints
  agent/
    graph.py            # LangGraph state machine
    nodes.py            # Agent node functions
    state.py            # AgentState schema
  services/
    stt.py              # Deepgram STT integration
    tts.py              # ElevenLabs TTS integration
    llm.py              # OpenAI LLM calls
  slides/
    content.py          # Slide data definitions
  config.py             # Settings via pydantic-settings
tests/
```

## Key Patterns

- Type hints on all function signatures
- Docstrings for public functions
- `raise` specific exceptions, not generic `Exception`
- Use `pathlib.Path` over `os.path`
- Prefer `dataclasses` or Pydantic models over raw dicts
- Virtual environments for isolation

## Testing

```bash
/python-review       # Python code review
/tdd                 # TDD workflow
/test-coverage       # Coverage analysis
```

## Git Workflow

- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`
- Feature branches from `main`

## Environment Variables

```bash
# Required
OPENAI_API_KEY=
DEEPGRAM_API_KEY=
ELEVENLABS_API_KEY=

# Optional
ELEVENLABS_VOICE_ID=   # defaults to a preset voice
DEBUG=false
```
