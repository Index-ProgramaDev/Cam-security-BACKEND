"""
Functional test for pipeline.py corrections.
Tests run without YOLO/Grad-CAM to avoid download latency — those components
are unchanged and were confirmed working before.

Tests:
  T1. Import and checkpoint load
  T2. _extract_frames_gif — correct FPS, frame count, no padding
  T3. _extract_frames on a real video file (if available)
  T4. analyze() on fighting.gif — correct duration, no padding artifacts
  T5. analyze() on a <CLIP_LENGTH synthetic video — returns empty result + warning
  T6. verbose flag — diag lines appear on verbose=True, absent on False
"""
import sys, io, types, contextlib
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append(condition)
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


# ── T1: import + checkpoint ────────────────────────────────────────────────────
print("\n── T1: Import and checkpoint load ──")
try:
    from src.inference.pipeline import SurveillancePipeline, get_pipeline
    from config import CLIP_LENGTH, FRAME_INTERVAL
    check("T1.1 import pipeline", True)
except Exception as e:
    check("T1.1 import pipeline", False, str(e))
    print("Cannot continue — import failed."); sys.exit(1)

# Load pipeline but stub out YOLO + GradCAM to avoid slow downloads
import unittest.mock as mock
import torch, numpy as np
from PIL import Image

with mock.patch("src.inference.pipeline.get_detector") as mock_det, \
     mock.patch("src.inference.pipeline.GradCAM") as mock_gc:
    dummy_heatmap = np.zeros((64, 64), dtype=np.float32)
    dummy_overlay = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    mock_det_inst = mock.MagicMock()
    mock_det_inst.detect.return_value = []
    mock_det_inst.summarize.return_value = ""
    mock_det.return_value = mock_det_inst
    mock_gc_inst = mock.MagicMock()
    mock_gc_inst.compute_and_overlay.return_value = (dummy_heatmap, dummy_overlay)
    mock_gc.return_value = mock_gc_inst
    try:
        pl = get_pipeline(verbose=True)
        check("T1.2 checkpoint load", True,
              f"epoch={pl.lstm.__class__.__name__} loaded OK")
    except Exception as e:
        check("T1.2 checkpoint load", False, str(e))
        sys.exit(1)


# ── T2: _extract_frames_gif ────────────────────────────────────────────────────
print("\n── T2: GIF frame extraction ──")
gif_path = Path("fighting.gif")
if gif_path.exists():
    frames, fps, n_raw = pl._extract_frames_gif(str(gif_path))

    # fighting.gif: 71 frames, 10 fps, 7.1s
    check("T2.1 n_raw == 71",       n_raw == 71,            f"got {n_raw}")
    check("T2.2 fps == 10.0",       abs(fps - 10.0) < 0.01, f"got {fps:.4f}")

    # 71 sampled at FRAME_INTERVAL=10 → 7 < CLIP_LENGTH=16 → fallback to all 71
    check("T2.3 fallback: 71 frames returned", len(frames) == 71,
          f"got {len(frames)}")

    # Duration = 71 / 10.0 = 7.1s  (was 71s before fix)
    duration = len(frames) / fps
    check("T2.4 duration ~7.1s",    abs(duration - 7.1) < 0.2, f"got {duration:.2f}s")

    # With 71 frames and fallback fps=10.0: (71-16)//8 + 1 = 7 clips, all real
    expected_clips = (len(frames) - CLIP_LENGTH) // 8 + 1
    check("T2.5 enough frames for real clips (no padding needed)",
          len(frames) >= CLIP_LENGTH,
          f"{len(frames)} >= {CLIP_LENGTH}")
    print(f"       → would produce {expected_clips} clips with 16 real frames each")
else:
    print("  [SKIP] fighting.gif not found")


# ── T3: _extract_frames on real video ─────────────────────────────────────────
print("\n── T3: Regular video extraction ──")
video_path = Path("data/test_video.mp4")
if video_path.exists():
    frames_v, fps_v, total_v = pl._extract_frames(str(video_path))
    check("T3.1 frames extracted",   len(frames_v) > 0,    f"{len(frames_v)} frames")
    check("T3.2 fps > 0",            fps_v > 0,             f"{fps_v:.1f} fps")
    check("T3.3 PIL Images returned",
          all(isinstance(f, Image.Image) for f in frames_v[:3]), "")
    # Effective fps should be original_fps / FRAME_INTERVAL (unless fallback)
    print(f"       → {len(frames_v)} frames @ {fps_v:.2f} fps effective")
else:
    print("  [SKIP] data/test_video.mp4 not found")


# ── T4: analyze() on fighting.gif ─────────────────────────────────────────────
print("\n── T4: analyze() on fighting.gif ──")
if gif_path.exists():
    # Suppress verbose output from diag lines for clean test output
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = pl.analyze(str(gif_path))
    output = buf.getvalue()

    check("T4.1 clips_analyzed > 0",  result.clips_analyzed > 0,
          f"got {result.clips_analyzed}")
    check("T4.2 duration ~7.1s",      abs(result.duration_sec - 7.1) < 0.5,
          f"got {result.duration_sec:.2f}s  (was 71s before fix)")
    check("T4.3 fps == 10.0",         abs(result.fps - 10.0) < 0.01,
          f"got {result.fps:.2f}")

    # The key fix: score should now reflect real content, not padding artifact
    # (we can't assert the exact value, but we can verify scores are in [0,1])
    all_scores = [r.anomaly_score for r in result.results]
    check("T4.4 all scores in [0,1]",
          all(0.0 <= s <= 1.0 for s in all_scores),
          f"scores: {[f'{s:.4f}' for s in all_scores]}")
    check("T4.5 no VideoResult with empty result from padding",
          result.clips_analyzed > 0,  # should have real clips now
          f"clips={result.clips_analyzed}")
    print(f"       → scores: {[f'{s:.4f}' for s in all_scores]}")
    print(f"       → anomalous: {result.anomalous_clips}/{result.clips_analyzed}")
else:
    print("  [SKIP] fighting.gif not found")


# ── T5: Short video returns empty result, no padding ──────────────────────────
print("\n── T5: Short video (< CLIP_LENGTH frames) handling ──")

# Create a tiny synthetic GIF with 4 frames (< CLIP_LENGTH=16)
import tempfile, os
tiny_frames = [Image.fromarray(np.random.randint(0, 255, (64,64,3), dtype=np.uint8))
               for _ in range(4)]
with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
    tmp_path = tmp.name

tiny_frames[0].save(
    tmp_path,
    save_all=True,
    append_images=tiny_frames[1:],
    duration=100,
    loop=0,
)

buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    result_short = pl.analyze(tmp_path)
out2 = buf2.getvalue()
os.unlink(tmp_path)

check("T5.1 clips_analyzed == 0 (no padding)",
      result_short.clips_analyzed == 0,
      f"got {result_short.clips_analyzed}")
check("T5.2 warning printed",
      "WARNING" in out2 or "too short" in out2.lower(),
      "")
check("T5.3 max_score == 0.0",
      result_short.max_score == 0.0,
      f"got {result_short.max_score}")
check("T5.4 results list is empty",
      result_short.results == [],
      f"got {len(result_short.results)} items")


# ── T6: verbose flag ───────────────────────────────────────────────────────────
print("\n── T6: verbose flag ──")
if gif_path.exists():
    # verbose=True — diag lines must appear
    pl.verbose = True
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        _ = pl.analyze(str(gif_path))
    out3 = buf3.getvalue()
    check("T6.1 [diag] lines present when verbose=True", "[diag]" in out3, "")

    # verbose=False — no diag lines
    pl.verbose = False
    buf4 = io.StringIO()
    with contextlib.redirect_stdout(buf4):
        _ = pl.analyze(str(gif_path))
    out4 = buf4.getvalue()
    check("T6.2 no [diag] lines when verbose=False", "[diag]" not in out4, "")
else:
    print("  [SKIP] fighting.gif not found")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
passed = sum(results)
total  = len(results)
print(f"RESULT: {passed}/{total} tests passed")
if passed == total:
    print("\033[32mAll tests passed.\033[0m")
else:
    print(f"\033[31m{total - passed} test(s) failed.\033[0m")
    sys.exit(1)
