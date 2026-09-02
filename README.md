# transcriber-gpu

GPU worker image for meeting transcription: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
large-v3 plus [pyannote speaker diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
The worker claims jobs from a queue server, streams timestamped lines back with a voice cluster on
each, posts the voice centroids at the end, and exits when idle.

Image: `ghcr.io/domainersuite/transcriber-gpu:latest`, built by `.github/workflows/publish-image.yml`.
Only `worker.py` and `entrypoint.sh` enter the image (`.dockerignore`); it holds no credentials,
audio or data. pyannote's gated models are fetched at start with `HF_TOKEN`.

Env: `TRANSCRIBER_SERVER`, `TRANSCRIBER_TOKEN` (required), `HF_TOKEN`, `WORKER_NAME`,
`WHISPER_MODEL` / `WHISPER_DEVICE` / `WHISPER_COMPUTE` / `WHISPER_BEAM`, `TRANSCRIBER_PASS`,
`TRANSCRIBER_DIARIZE`, `TRANSCRIBER_IDLE_EXIT`, `TRANSCRIBER_SLICE_S`.

`worker.py` is a copy of the one in the site repo (`scripts/trust-transcripts/worker.py`); when that
changes, copy it here and push — the workflow rebuilds the image.
