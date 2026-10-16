"""Inspeciona os pesos e biases do classificador sem modificá-los."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from src.models.lstm_model import AnomalyLSTM
from config import CHECKPOINTS_DIR

ckpt  = torch.load(CHECKPOINTS_DIR / "best_model.pt", map_location="cpu", weights_only=False)
model = AnomalyLSTM()
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"Checkpoint: epoch {ckpt['epoch']}, val_auc={ckpt['val_auc']:.4f}\n")

# classifier = [Dropout(0), Linear256→64(1), ReLU(2), Linear64→1(3), Sigmoid(4)]
lin256 = model.classifier[1]  # Linear(256, 64)
lin1   = model.classifier[3]  # Linear(64, 1)

print("── Linear(256→64) ──────────────────────────────────────")
print(f"  weight: shape={lin256.weight.shape}  "
      f"min={lin256.weight.min():.4f}  max={lin256.weight.max():.4f}  "
      f"mean={lin256.weight.mean():.4f}  std={lin256.weight.std():.4f}")
print(f"  bias  : shape={lin256.bias.shape}  "
      f"min={lin256.bias.min():.4f}  max={lin256.bias.max():.4f}  "
      f"mean={lin256.bias.mean():.4f}  std={lin256.bias.std():.4f}")

print("\n── Linear(64→1) ─────────────────────────────────────────")
print(f"  weight: shape={lin1.weight.shape}  "
      f"min={lin1.weight.min():.4f}  max={lin1.weight.max():.4f}  "
      f"mean={lin1.weight.mean():.4f}  std={lin1.weight.std():.4f}")
print(f"  bias  : {lin1.bias.item():.6f}   ← este valor domina o logit")

w = lin1.weight.detach().numpy().flatten()
b = lin1.bias.item()
print(f"\n── Análise do logit esperado ────────────────────────────")
print(f"  logit = w · relu_out + bias")
print(f"  Com relu_mean≈2.2 (observado nos clips):")
relu_mean_obs = 2.2
logit_estimate = float(np.sum(w) * relu_mean_obs) + b
print(f"    Σw = {np.sum(w):.4f}")
print(f"    Σw × relu_mean = {np.sum(w) * relu_mean_obs:.4f}")
print(f"    + bias = {b:.4f}")
print(f"    ≈ logit {logit_estimate:.4f}   (observado: ~37–41)")

print(f"\n── LSTM biases ──────────────────────────────────────────")
for name, p in model.named_parameters():
    if "bias" in name:
        print(f"  {name:<35} mean={p.mean():.4f}  std={p.std():.4f}  "
              f"norm={p.norm():.4f}")

print(f"\n── Diagnóstico ──────────────────────────────────────────")
print(f"  bias do Linear(64→1) = {b:.4f}")
if b > 10:
    print(f"  *** BIAS MUITO POSITIVO: mesmo com relu_out ≈ 0, logit seria {b:.1f}")
    print(f"      Isso significa que o modelo retorna score ≈ 1.0 para qualquer entrada.")
    print(f"      O classificador está colapsado — o bias domina completamente a saída.")
elif b < -10:
    print(f"  *** BIAS MUITO NEGATIVO: mesmo com relu_out > 0, logit tende a {b:.1f}")
    print(f"      O modelo retornaria score ≈ 0.0 para quase qualquer entrada.")
else:
    print(f"  Bias dentro de range razoável. O problema deve estar nos pesos ou nas features.")
