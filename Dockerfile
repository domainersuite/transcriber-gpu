# GPU transcription worker: whisper large-v3 + pyannote speaker diarization, for a rented GPU pod.
# Built by .github/workflows/publish-image.yml to ghcr.io/domainersuite/transcriber-gpu (public image).
# Runtime env (set by whoever launches the pod): TRANSCRIBER_SERVER, TRANSCRIBER_TOKEN, HF_TOKEN.
# The image holds no credentials, audio or data: only worker.py, entrypoint.sh and public models.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
# ctranslate2 (faster-whisper) dlopens cuBLAS 12 and cuDNN 9; the pip wheels are the documented way
# to provide them, and entrypoint.sh puts their lib dirs on LD_LIBRARY_PATH.
# huggingface_hub 1.x dropped the use_auth_token kwarg that pyannote 3.3 still passes; pin below 1.0.
RUN pip install --no-cache-dir "faster-whisper==1.*" "pyannote.audio==3.3.2" "huggingface_hub<1.0" \
      "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"
WORKDIR /app
COPY worker.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
# Bake the whisper model so a fresh pod starts transcribing within a minute. pyannote's models are
# gated and fetched at start with HF_TOKEN (a few hundred MB).
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"
ENV WHISPER_MODEL=large-v3 WHISPER_DEVICE=cuda WHISPER_COMPUTE=float16 WHISPER_BEAM=5 \
    TRANSCRIBER_PASS=any TRANSCRIBER_DIARIZE=1 TRANSCRIBER_IDLE_EXIT=600 TRANSCRIBER_SLICE_S=1800 \
    TRANSCRIBER_SERVER=https://abolishislandstrust.org
CMD ["/app/entrypoint.sh"]
