"""Vision pass: who is on screen when a voice speaks.

Zoom draws every participant's name on their tile and outlines the active speaker in green. This
module samples the Trust's video lightly (one 10-second HLS segment per minute, so ~1/6 of the
bytes), pulls a few frames from each, finds the green-outlined tile, reads the name tag along its
bottom edge with tesseract, and returns sightings: (seconds, text). It never decides who anybody
is: the server keeps a sighting only when the text names an official of that meeting.

Requires: ffmpeg, tesseract (apt tesseract-ocr), pillow, pytesseract.
"""
import concurrent.futures, os, subprocess, tempfile, time, urllib.parse, urllib.request

SAMPLE_EVERY = int(os.environ.get("VISION_SAMPLE_EVERY", "6"))     # take every Nth ~10 s segment
FRAMES_PER_SEGMENT = int(os.environ.get("VISION_FRAMES", "3"))
PARALLEL = int(os.environ.get("VISION_PARALLEL", "4"))

def _lines(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return [l.strip() for l in r.read().decode("utf-8", "replace").splitlines() if l.strip() and not l.startswith("#")]

def _durations(url):
    """(segment url, duration) pairs from a media playlist."""
    with urllib.request.urlopen(url, timeout=60) as r:
        text = r.read().decode("utf-8", "replace")
    out, dur = [], 10.0
    for l in text.splitlines():
        l = l.strip()
        if l.startswith("#EXTINF:"):
            try: dur = float(l[8:].split(",")[0])
            except ValueError: pass
        elif l and not l.startswith("#"):
            out.append((urllib.parse.urljoin(url, l), dur))
    return out

def video_segments(playlist_url):
    first = _lines(playlist_url)
    if not first: raise RuntimeError("empty playlist")
    media = urllib.parse.urljoin(playlist_url, first[0]) if first[0].split("?")[0].endswith(".m3u8") else playlist_url
    return _durations(media)

def _get(url, dest):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            return dest
        except Exception:
            if attempt == 3: raise
            time.sleep(1 + attempt)

def _green(px):
    r, g, b = px[:3]
    return g > 150 and r < 110 and b < 130 and g - max(r, b) > 60

def active_tile(im):
    """Bounding box of the green-outlined tile, or None."""
    w, h = im.size; px = im.load()
    rows = [y for y in range(0, h, 2) if sum(1 for x in range(0, w, 3) if _green(px[x, y])) > (w / 3) * 0.12]
    cols = [x for x in range(0, w, 2) if sum(1 for y in range(0, h, 3) if _green(px[x, y])) > (h / 3) * 0.12]
    if len(rows) < 2 or len(cols) < 2: return None
    box = (min(cols), min(rows), max(cols), max(rows))
    if box[2] - box[0] < 120 or box[3] - box[1] < 80: return None
    return box

def read_tag(im, box):
    """OCR the name tag along the bottom of the tile; returns cleaned text or ''."""
    import pytesseract
    x0, y0, x1, y1 = box
    strip = im.crop((x0, max(y0, y1 - 44), x1, y1 - 2))
    strip = strip.resize((strip.width * 3, strip.height * 3))
    text = pytesseract.image_to_string(strip, config="--psm 7")
    text = " ".join(text.split())
    # keep letters, apostrophes, hyphens, spaces; drop OCR noise
    import re
    text = re.sub(r"[^A-Za-z'’\- ]+", " ", text)
    text = " ".join(w for w in text.split() if len(w) > 1)
    return text[:80]

def frames_from_segment(ffmpeg, path, n):
    outdir = tempfile.mkdtemp()
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", path, "-vf", f"fps={n}/10", "-frames:v", str(n), os.path.join(outdir, "f%02d.png")], check=False)
    return sorted(os.path.join(outdir, f) for f in os.listdir(outdir))

def sightings(playlist_url, ffmpeg, log, heartbeat=None):
    """Yield (t_seconds, text) for every sampled frame whose outlined tile carries a readable tag."""
    from PIL import Image
    segs = video_segments(playlist_url)
    picks, t = [], 0.0
    for i, (url, dur) in enumerate(segs):
        if i % SAMPLE_EVERY == 0: picks.append((t, url, dur))
        t += dur
    log(f"vision: {len(segs)} segments, sampling {len(picks)}")
    out, done, beat = [], 0, time.time()
    with tempfile.TemporaryDirectory() as td, concurrent.futures.ThreadPoolExecutor(PARALLEL) as ex:
        def work(item):
            t0, url, dur = item
            seg = _get(url, os.path.join(td, f"{int(t0)}.ts"))
            res = []
            for k, fp in enumerate(frames_from_segment(ffmpeg, seg, FRAMES_PER_SEGMENT)):
                try:
                    im = Image.open(fp).convert("RGB")
                    box = active_tile(im)
                    if box:
                        text = read_tag(im, box)
                        if len(text) >= 4: res.append((round(t0 + k * (10.0 / FRAMES_PER_SEGMENT), 1), text))
                finally:
                    try: os.remove(fp)
                    except OSError: pass
            try: os.remove(seg)
            except OSError: pass
            return res
        for res in ex.map(work, picks):
            out.extend(res); done += 1
            if heartbeat and time.time() - beat > 60:
                try: heartbeat(done, len(picks))
                except Exception as e: log(f"heartbeat failed: {e}")
                beat = time.time()
    log(f"vision: {len(out)} sightings from {len(picks)} sampled segments")
    return out
