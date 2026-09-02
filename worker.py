"""Transcription worker — claims jobs from the abolishislandstrust.org server, transcribes, streams segments back.

The server owns the queue and the data (migrations/0108 + 0109, server/meetingTranscripts.ts). This
process only needs: the server URL, TRANSCRIBER_TOKEN, ffmpeg, and faster-whisper. It runs
unchanged on the droplet (compose service `transcriber`, small.en, nice 19), on a laptop
(medium.en) or on a rented GPU pod (ghcr.io/domainersuite/transcriber-gpu: large-v3 + pyannote diarization) — whichever
claims a job first does it.

  python worker.py --server https://abolishislandstrust.org --model small.en --pass fast --loop
  python worker.py --server https://abolishislandstrust.org --model medium.en --pass final --once
  python worker.py --server https://abolishislandstrust.org --model large-v3 --device cuda --compute-type float16
                   --pass any --diarize --loop --idle-exit 600      # what the RunPod pod runs

Who said what: with --diarize the worker first runs pyannote/speaker-diarization-3.1 over the whole
recording (needs HF_TOKEN; accept the model terms on Hugging Face once), then transcribes with word
timestamps and splits every whisper segment wherever the voice changes. Each segment is posted with
its VOICE CLUSTER ("SPEAKER_03"); the voice centroids go up at the end so the server can match them
against saved voiceprints and an admin can name each voice once. Nothing here names anybody.

Env: TRANSCRIBER_TOKEN (required), WORKER_NAME (default: RUNPOD_POD_ID, else hostname), FFMPEG
(optional path), HF_TOKEN (for --diarize), WHISPER_MODEL / WHISPER_DEVICE / WHISPER_COMPUTE /
WHISPER_THREADS / TRANSCRIBER_PASS / TRANSCRIBER_DIARIZE / TRANSCRIBER_IDLE_EXIT (CLI defaults).
"""
import argparse, bisect, json, os, socket, subprocess, sys, tempfile, time, types, urllib.request, urllib.error

def http(server, token, method, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(server.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "x-worker-token": token,
                                          "User-Agent": "trust-transcriber/2"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code}: {e.read()[:300]!r}")

def ffmpeg_exe():
    if os.environ.get("FFMPEG"): return os.environ["FFMPEG"]
    for cand in ("ffmpeg",):
        try:
            subprocess.run([cand, "-version"], capture_output=True, check=True); return cand
        except Exception: pass
    import imageio_ffmpeg  # pip install imageio-ffmpeg (Windows box without a system ffmpeg)
    return imageio_ffmpeg.get_ffmpeg_exe()

def fetch_wav(stream_url, out, heartbeat=None):
    """16 kHz mono WAV from the HLS stream. Wowza serves an audio-only rendition of the same
    recording (?wowzaaudioonly) that is a fraction of the size of the 720p video; try it first and
    fall back to the video playlist if the host does not offer it."""
    urls = [stream_url]
    if "playlist.m3u8" in stream_url and "?" not in stream_url:
        urls.insert(0, stream_url + "?wowzaaudioonly")
    last = None
    for u in urls:
        proc = subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-y", "-i", u,
                                 "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out])
        beat = time.time()
        while proc.poll() is None:
            time.sleep(5)
            # The host serves HLS at only a few times real time; a long meeting takes many minutes
            # to fetch, and a silent worker is requeued after 30. Say we are alive while we wait.
            if heartbeat and time.time() - beat > 60:
                try: heartbeat(os.path.getsize(out) if os.path.exists(out) else 0)
                except Exception as e: print("heartbeat during fetch failed:", e, flush=True)
                beat = time.time()
        if proc.returncode == 0 and os.path.getsize(out) > 1_000_000: return
        last = subprocess.CalledProcessError(proc.returncode, "ffmpeg")
    if last: raise last

def wav_duration(path):
    import wave
    w = wave.open(path, "rb")
    assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    d = w.getnframes() / 16000
    w.close()
    return d

def load_wav_slice(path, start_s, end_s):
    """Read only [start_s, end_s). A whole 5-hour meeting is ~1.1 GB as float32, which is more
    than the droplet container may hold; 20-minute slices are ~77 MB."""
    import wave, numpy as np
    w = wave.open(path, "rb")
    total = w.getnframes()
    first, last = min(total, int(start_s * 16000)), min(total, int(end_s * 16000))
    w.setpos(first)
    a = np.frombuffer(w.readframes(max(0, last - first)), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    return a

SLICE_S = int(os.environ.get("TRANSCRIBER_SLICE_S", 20 * 60))

# ── diarization ──────────────────────────────────────────────────────────────

class Voices:
    """Speaker turns from pyannote, with a lookup that returns the voice speaking in [a, b)."""
    def __init__(self, turns, centroids):
        self.turns = sorted(turns)                       # (start, end, label)
        self.starts = [t[0] for t in self.turns]
        self.centroids = centroids                       # label -> list[float]
        self.talk = {}
        self.longest = {}
        for s, e, l in self.turns:
            self.talk[l] = self.talk.get(l, 0.0) + (e - s)
            if e - s > self.longest.get(l, (0, 0, 0))[2]: self.longest[l] = (s, e, e - s)

    def at(self, a, b, tolerance=1.0):
        best, best_ov = None, 0.0
        hi = bisect.bisect_left(self.starts, b)
        for k in range(hi - 1, max(-1, hi - 400), -1):   # turns are sorted by start; scan back for overlaps
            s, e, l = self.turns[k]
            ov = min(e, b) - max(s, a)
            if ov > best_ov: best, best_ov = l, ov
            if s < a - 3600: break
        if best is not None: return best
        # No overlap (a gap between turns): take the nearest turn edge within `tolerance` seconds.
        mid = (a + b) / 2
        near, near_d = None, tolerance
        for k in range(max(0, hi - 3), min(len(self.turns), hi + 3)):
            s, e, l = self.turns[k]
            d = 0.0 if s <= mid <= e else min(abs(s - mid), abs(e - mid))
            if d < near_d: near, near_d = l, d
        return near

def diarize(wav, duration, device, log):
    import torch
    from pyannote.audio import Pipeline
    token = os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("--diarize needs HF_TOKEN (accept pyannote/segmentation-3.0 and pyannote/speaker-diarization-3.1 on Hugging Face)")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    pipe.to(torch.device(device))
    audio = load_wav_slice(wav, 0, duration)
    waveform = torch.from_numpy(audio)[None]
    t0 = time.time()
    diar, centroids = pipe({"waveform": waveform, "sample_rate": 16000}, return_embeddings=True)
    labels = list(diar.labels())
    turns = [(float(seg.start), float(seg.end), str(label)) for seg, _, label in diar.itertracks(yield_label=True)]
    cents = {label: [float(x) for x in centroids[i]] for i, label in enumerate(labels)} if centroids is not None else {}
    log(f"diarization: {len(labels)} voices, {len(turns)} turns in {time.time() - t0:.0f}s")
    del waveform, audio, pipe
    if device.startswith("cuda"): torch.cuda.empty_cache()
    return Voices(turns, cents)

MIN_RUN_S = 0.8   # a voice change shorter than this inside one whisper segment is noise, not a speaker

def split_by_voice(seg, offset, voices):
    """One whisper segment -> [(start, end, text, cluster)], split wherever the voice changes."""
    words = getattr(seg, "words", None) or []
    if not words:
        a, b = seg.start + offset, seg.end + offset
        return [(a, b, seg.text.strip(), voices.at(a, b))]
    runs = []   # [start, end, [words], cluster]
    for w in words:
        a, b = w.start + offset, w.end + offset
        c = voices.at(a, b)
        if runs and (runs[-1][3] == c or c is None):
            runs[-1][1] = b; runs[-1][2].append(w.word)
        else:
            runs.append([a, b, [w.word], c])
    # merge runs too short to be a real turn into the previous run
    merged = []
    for r in runs:
        if merged and (r[1] - r[0]) < MIN_RUN_S and len(r[2]) <= 2:
            merged[-1][1] = r[1]; merged[-1][2].extend(r[2])
        else:
            merged.append(r)
    return [(r[0], r[1], "".join(r[2]).strip(), r[3]) for r in merged if "".join(r[2]).strip()]

# ── one job ──────────────────────────────────────────────────────────────────

def transcribe(job, args, server, token):
    rec = job["recording"]
    log = lambda m: print(f"[{job['id']}] {m}", flush=True)
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, rec["recordingId"] + ".wav")
        log(f"fetching {rec['recordingId']} …")
        fetch_wav(rec["streamUrl"], wav, heartbeat=lambda nbytes: http(server, token, "POST",
                  f"/api/transcriber/jobs/{job['id']}/heartbeat", {"progressS": 0, "fetchedBytes": nbytes}))
        duration = wav_duration(wav)
        log(f"{duration/3600:.2f} h of audio; model {args.model} on {args.device}; diarize={args.diarize}")
        http(server, token, "POST", f"/api/transcriber/jobs/{job['id']}/heartbeat",
             {"durationS": int(duration), "progressS": 0, "model": args.model})

        voices = diarize(wav, duration, args.device, log) if args.diarize else None
        if voices:
            http(server, token, "POST", f"/api/transcriber/jobs/{job['id']}/heartbeat", {"progressS": 0})

        sys.modules.setdefault("av", types.ModuleType("av"))  # av's DLL is blocked on the Windows box; unused here
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type, cpu_threads=args.threads)
        batch, idx, last_flush = [], 0, time.time()
        seg_count = {}
        def flush(final=False):
            nonlocal batch, last_flush
            if not batch and not final: return
            http(server, token, "POST", f"/api/transcriber/jobs/{job['id']}/segments",
                 {"segments": batch, "progressS": int(batch[-1]["end"]) if batch else None})
            batch, last_flush = [], time.time()
        offset = 0.0
        while offset < duration:
            audio = load_wav_slice(wav, offset, min(duration, offset + SLICE_S))
            segs, _info = model.transcribe(audio, beam_size=args.beam, vad_filter=True,
                                           vad_parameters=dict(min_silence_duration_ms=700),
                                           condition_on_previous_text=False, word_timestamps=bool(voices))
            for s in segs:
                pieces = split_by_voice(s, offset, voices) if voices else [(s.start + offset, s.end + offset, s.text.strip(), None)]
                for a, b, text, cluster in pieces:
                    if not text: continue
                    row = {"idx": idx, "start": round(a, 2), "end": round(b, 2), "text": text}
                    if cluster: row["cluster"] = cluster; seg_count[cluster] = seg_count.get(cluster, 0) + 1
                    batch.append(row); idx += 1
                    if len(batch) >= 40 or time.time() - last_flush > 60: flush()
            flush()
            http(server, token, "POST", f"/api/transcriber/jobs/{job['id']}/heartbeat",
                 {"progressS": int(min(duration, offset + SLICE_S))})
            offset += SLICE_S
        flush()
        if voices:
            clusters = []
            for label in sorted(voices.talk, key=lambda l: -voices.talk[l]):
                s, e, _ = voices.longest.get(label, (None, None, 0))
                clusters.append({"cluster": label, "talkS": round(voices.talk[label], 2), "segmentCount": seg_count.get(label, 0),
                                 "embedding": voices.centroids.get(label), "sampleStart": s, "sampleEnd": None if e is None else min(e, s + 90)})
            http(server, token, "POST", f"/api/transcriber/jobs/{job['id']}/speakers",
                 {"clusters": clusters, "model": "pyannote/speaker-diarization-3.1"})
        http(server, token, "POST", f"/api/transcriber/jobs/{job['id']}/complete",
             {"durationS": int(duration), "segmentCount": idx, "model": args.model})
        log(f"done: {idx} segments")

def main():
    env = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=env("TRANSCRIBER_SERVER", "http://localhost:5000"))
    ap.add_argument("--model", default=env("WHISPER_MODEL", "small.en"))
    ap.add_argument("--pass", dest="pass_", default=env("TRANSCRIBER_PASS", "fast"), choices=["fast", "final", "any"])
    ap.add_argument("--threads", type=int, default=int(env("WHISPER_THREADS", "2")))
    ap.add_argument("--beam", type=int, default=int(env("WHISPER_BEAM", "1")))
    ap.add_argument("--device", default=env("WHISPER_DEVICE", "cpu")); ap.add_argument("--compute-type", default=env("WHISPER_COMPUTE", "int8"))
    ap.add_argument("--diarize", action="store_true", default=env("TRANSCRIBER_DIARIZE", "") in ("1", "true", "yes"))
    ap.add_argument("--loop", action="store_true"); ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--idle-exit", type=int, default=int(env("TRANSCRIBER_IDLE_EXIT", "0")),
                    help="with --loop: after this many seconds without a job, tell the server we are idle and exit (a rented pod)")
    args = ap.parse_args()
    token = env("TRANSCRIBER_TOKEN")
    if not token: sys.exit("TRANSCRIBER_TOKEN is required")
    worker = env("WORKER_NAME") or env("RUNPOD_POD_ID") or socket.gethostname()
    last_work = time.time()
    while True:
        job = None
        try:
            job = http(args.server, token, "POST", "/api/transcriber/claim",
                       {"worker": worker, "model": args.model, "pass": args.pass_, "diarize": args.diarize})
        except Exception as e:
            print("claim failed:", e, flush=True)
        if job:
            last_work = time.time()
            try:
                transcribe(job, args, args.server, token)
            except Exception as e:
                print(f"[{job['id']}] FAILED: {e}", flush=True)
                try: http(args.server, token, "POST", f"/api/transcriber/jobs/{job['id']}/fail", {"error": str(e)[:2000]})
                except Exception as e2: print("could not report failure:", e2, flush=True)
            last_work = time.time()
            if args.once: return
            continue
        if not args.loop: return
        if args.idle_exit and time.time() - last_work > args.idle_exit:
            print(f"idle for {args.idle_exit}s; reporting idle and exiting", flush=True)
            try: http(args.server, token, "POST", "/api/transcriber/idle", {"worker": worker})
            except Exception as e: print("idle report failed:", e, flush=True)
            return
        time.sleep(args.poll)

if __name__ == "__main__":
    main()
