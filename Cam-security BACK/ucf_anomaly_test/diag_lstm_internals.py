"""
Diagnóstico interno da LSTM — sem modificar pesos, arquitetura ou threshold.

Inspeciona, para cada clip:
  1. hidden[-1]          — estado oculto da camada LSTM superior
  2. classifier[1] out   — Linear(256→64)
  3. classifier[2] out   — ReLU
  4. classifier[3] out   — logit (Linear 64→1), antes da Sigmoid
  5. score               — após Sigmoid

Roda em fighting.gif e data/test_video.mp4 com o mesmo checkpoint e
pré-processamento da pipeline.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

from config import (
    DEVICE, CLIP_LENGTH, CLIP_STRIDE, FRAME_SIZE,
    CHECKPOINTS_DIR, FRAME_INTERVAL, FEATURE_DIM
)
from src.models.feature_extractor import get_extractor
from src.models.lstm_model import AnomalyLSTM

# ── mesma normalização da pipeline e do extract_features.py ──────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── carga do modelo ────────────────────────────────────────────────────────────
ckpt = torch.load(CHECKPOINTS_DIR / "best_model.pt",
                  map_location=DEVICE, weights_only=False)
model = AnomalyLSTM().to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

extractor = get_extractor(DEVICE)


# ── extração de frames — replica _extract_frames_gif e _extract_frames ────────

def extract_frames_gif(path: str):
    gif = Image.open(path)
    raw, durs = [], []
    try:
        while True:
            raw.append(gif.copy().convert("RGB"))
            durs.append(gif.info.get("duration", 100))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    n = len(raw)
    avg_ms  = sum(durs) / n if n else 100.0
    gif_fps = 1000.0 / avg_ms
    sampled = [f for i, f in enumerate(raw) if i % FRAME_INTERVAL == 0]
    if len(sampled) < CLIP_LENGTH:          # fallback — usa todos
        return raw, gif_fps, n
    return sampled, gif_fps / FRAME_INTERVAL, n


def extract_frames_video(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % FRAME_INTERVAL == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    if len(frames) < CLIP_LENGTH and idx > 0:   # fallback
        cap2 = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap2.release()
        return frames, fps, idx
    return frames, fps / FRAME_INTERVAL, total


def frames_to_features(frames):
    all_f = []
    for i in range(0, len(frames), 64):
        batch = torch.stack([TRANSFORM(f) for f in frames[i:i+64]]).to(DEVICE)
        with torch.no_grad():
            all_f.append(extractor(batch).cpu().numpy())
    return np.concatenate(all_f, axis=0)


def build_clips(features):
    clips = []
    N = len(features)
    for start in range(0, N - CLIP_LENGTH + 1, CLIP_STRIDE):
        clips.append(features[start:start + CLIP_LENGTH])
    return clips  # list of [16, 2048]


# ── diagnóstico interno: hook manual passo a passo ────────────────────────────

def inspect_clips(clips, label):
    """
    Passa cada clip pelo modelo e coleta os estados internos.
    Não modifica pesos — usa apenas forward com torch.no_grad().
    """
    print(f"\n{'='*60}")
    print(f"  {label}  ({len(clips)} clips)")
    print(f"{'='*60}")

    hidden_norms = []
    lin64_means  = []
    relu_means   = []
    logits       = []
    scores       = []

    for ci, clip in enumerate(clips):
        x = torch.FloatTensor(clip).unsqueeze(0).to(DEVICE)  # [1, 16, 2048]

        with torch.no_grad():
            # ── LSTM ──────────────────────────────────────────────────────────
            _, (hidden, _) = model.lstm(x)
            h = hidden[-1]              # [1, 256]  — topo da pilha de LSTMs

            # ── Classifier passo a passo ──────────────────────────────────────
            # classifier = [Dropout(0), Linear(256→64)(1), ReLU(2), Linear(64→1)(3), Sigmoid(4)]
            after_dropout = model.classifier[0](h)      # Dropout (eval → identidade)
            after_lin64   = model.classifier[1](after_dropout)  # [1, 64]
            after_relu    = model.classifier[2](after_lin64)    # [1, 64]
            logit         = model.classifier[3](after_relu)     # [1, 1]
            score         = model.classifier[4](logit)          # [1, 1]  — Sigmoid

        h_norm   = h.norm().item()
        lin64_m  = after_lin64.mean().item()
        relu_m   = after_relu.mean().item()
        logit_v  = logit.item()
        score_v  = score.item()

        hidden_norms.append(h_norm)
        lin64_means.append(lin64_m)
        relu_means.append(relu_m)
        logits.append(logit_v)
        scores.append(score_v)

        print(f"  clip {ci:02d} │ "
              f"‖hidden[-1]‖={h_norm:7.3f} │ "
              f"lin64_mean={lin64_m:+7.4f} │ "
              f"relu_mean={relu_m:7.4f} │ "
              f"logit={logit_v:+9.4f} │ "
              f"score={score_v:.8f}")

    print(f"\n  Summary:")
    print(f"    hidden[-1] norm  : "
          f"min={min(hidden_norms):.3f}  max={max(hidden_norms):.3f}  "
          f"mean={np.mean(hidden_norms):.3f}")
    print(f"    lin64 mean       : "
          f"min={min(lin64_means):+.4f}  max={max(lin64_means):+.4f}  "
          f"mean={np.mean(lin64_means):+.4f}")
    print(f"    relu mean        : "
          f"min={min(relu_means):.4f}  max={max(relu_means):.4f}  "
          f"mean={np.mean(relu_means):.4f}")
    print(f"    logits           : "
          f"min={min(logits):+.4f}  max={max(logits):+.4f}  "
          f"mean={np.mean(logits):+.4f}  std={np.std(logits):.4f}")
    print(f"    scores           : "
          f"min={min(scores):.8f}  max={max(scores):.8f}  "
          f"mean={np.mean(scores):.8f}  std={np.std(scores):.8f}")
    print(f"    scores distinct? : "
          f"{len(set(f'{s:.10f}' for s in scores))} unique values out of {len(scores)}")
    print(f"    sigmoid(0)=0.5   : logit=0.0 | "
          f"score=1.0 ← logit > ~13.8  | score=0.0 ← logit < ~-13.8")
    return logits, scores


# ── main ──────────────────────────────────────────────────────────────────────

print("Loading extractor + LSTM checkpoint...")
print(f"  Checkpoint: epoch {ckpt['epoch']}, val AUC={ckpt['val_auc']:.4f}\n")

# ── fighting.gif ──────────────────────────────────────────────────────────────
gif_path = Path("fighting.gif")
if gif_path.exists():
    frames_g, fps_g, n_raw_g = extract_frames_gif(str(gif_path))
    print(f"fighting.gif  : {n_raw_g} raw frames → {len(frames_g)} extracted @ {fps_g:.1f}fps")
    feats_g  = frames_to_features(frames_g)
    clips_g  = build_clips(feats_g)
    logits_g, scores_g = inspect_clips(clips_g, "fighting.gif")
else:
    print("fighting.gif not found — skipping")
    logits_g, scores_g = [], []

# ── test_video.mp4 (video4.mp4 fallback) ──────────────────────────────────────
vid_candidates = [
    Path("data/test_video.mp4"),
    Path("../runs/detect/track/video4.avi"),
    Path("../runs/detect/track-2/video4.avi"),
]
vid_path = next((p for p in vid_candidates if p.exists()), None)

if vid_path:
    frames_v, fps_v, n_raw_v = extract_frames_video(str(vid_path))
    print(f"\n{vid_path.name}: {n_raw_v} raw frames → {len(frames_v)} extracted @ {fps_v:.1f}fps")
    feats_v  = frames_to_features(frames_v)
    clips_v  = build_clips(feats_v)
    if clips_v:
        logits_v, scores_v = inspect_clips(clips_v, str(vid_path.name))
    else:
        print(f"  Too few frames ({len(frames_v)}) — no clips formed")
        logits_v, scores_v = [], []
else:
    print("\nNo video file found — skipping video comparison")
    logits_v, scores_v = [], []

# ── Cross-video comparison ─────────────────────────────────────────────────────
if logits_g and logits_v:
    print(f"\n{'='*60}")
    print("  CROSS-VIDEO COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'fighting.gif':>18} {'video':>18}")
    print(f"  {'-'*66}")
    print(f"  {'logit mean':30} {np.mean(logits_g):>+18.4f} {np.mean(logits_v):>+18.4f}")
    print(f"  {'logit min':30} {min(logits_g):>+18.4f} {min(logits_v):>+18.4f}")
    print(f"  {'logit max':30} {max(logits_g):>+18.4f} {max(logits_v):>+18.4f}")
    print(f"  {'logit std':30} {np.std(logits_g):>18.4f} {np.std(logits_v):>18.4f}")
    print(f"  {'score mean':30} {np.mean(scores_g):>18.8f} {np.mean(scores_v):>18.8f}")
    print(f"  {'score min':30} {min(scores_g):>18.8f} {min(scores_v):>18.8f}")
    print(f"  {'score max':30} {max(scores_g):>18.8f} {max(scores_v):>18.8f}")

    sat_gif   = sum(1 for s in scores_g if s > 0.9999 or s < 0.0001)
    sat_vid   = sum(1 for s in scores_v if s > 0.9999 or s < 0.0001)
    print(f"\n  Saturated scores (>0.9999 or <0.0001):")
    print(f"    fighting.gif : {sat_gif}/{len(scores_g)}")
    print(f"    video        : {sat_vid}/{len(scores_v)}")
    print(f"\n  NOTE: Sigmoid saturates to 1.0 for logit > ~13.8")
    print(f"        Sigmoid saturates to 0.0 for logit < ~-13.8")
    if all(l > 13.8 for l in logits_g):
        print(f"\n  *** fighting.gif: ALL logits > 13.8 — Sigmoid is fully saturated.")
        print(f"      Scores are mathematically 1.0, not just rounded.")
        print(f"      The model is highly confident — not a precision artifact.")
    if all(l < -13.8 for l in logits_v):
        print(f"\n  *** video: ALL logits < -13.8 — Sigmoid is fully saturated at 0.")
        print(f"      The model is highly confident these clips are normal.")

print("\nDone.")
