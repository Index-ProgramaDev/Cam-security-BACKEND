"""
Decompõe o logit neurônio a neurônio para encontrar por que logit ≈ +37
mesmo com Σw = -3.1 e bias = 0.08.

logit = Σ_i (w_i * relu_i) + bias

A média de relu não é suficiente: o produto w_i*relu_i pode ser positivo mesmo
com Σw negativo, se os neurônios com w positivo tiverem activações muito altas
e os com w negativo tiverem activações próximas de zero (ReLU as zera).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

from config import DEVICE, CLIP_LENGTH, CLIP_STRIDE, FRAME_SIZE, CHECKPOINTS_DIR, FRAME_INTERVAL, FEATURE_DIM
from src.models.feature_extractor import get_extractor
from src.models.lstm_model import AnomalyLSTM

TRANSFORM = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ckpt  = torch.load(CHECKPOINTS_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
model = AnomalyLSTM().to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
extractor = get_extractor(DEVICE)

def get_one_clip_features(path, is_gif=False):
    if is_gif:
        gif = Image.open(path)
        frames = []
        try:
            while True:
                frames.append(gif.copy().convert("RGB"))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
    else:
        cap = cv2.VideoCapture(path)
        raw = []
        while True:
            ret, f = cap.read()
            if not ret: break
            raw.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        cap.release()
        frames = raw

    tensors = torch.stack([TRANSFORM(f) for f in frames[:64]]).to(DEVICE)
    with torch.no_grad():
        feats = extractor(tensors).cpu().numpy()

    # primeiro clip disponível
    clip = feats[:CLIP_LENGTH]
    return clip, frames[:CLIP_LENGTH]


def decompose_logit(clip_np, label):
    x = torch.FloatTensor(clip_np).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, (hidden, _) = model.lstm(x)
        h = hidden[-1]
        after_dropout = model.classifier[0](h)
        after_lin64   = model.classifier[1](after_dropout)   # [1, 64]
        after_relu    = model.classifier[2](after_lin64)     # [1, 64]
        logit         = model.classifier[3](after_relu)      # [1, 1]
        score         = model.classifier[4](logit)

    w  = model.classifier[3].weight.detach().cpu().numpy().flatten()  # [64]
    b  = model.classifier[3].bias.item()
    r  = after_relu.detach().cpu().numpy().flatten()  # [64]

    contributions = w * r   # [64]  — contribuição de cada neurônio ao logit

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  logit  = {logit.item():+.4f}")
    print(f"  score  = {score.item():.8f}")
    print(f"  bias   = {b:+.6f}")
    print(f"  Σ(w·relu) = {contributions.sum():.4f}  (= logit - bias = {logit.item()-b:.4f})")
    print(f"\n  relu activations — {(r > 0).sum()}/64 neurons active")
    print(f"    min={r.min():.4f}  max={r.max():.4f}  mean={r.mean():.4f}  std={r.std():.4f}")

    print(f"\n  Weights of Linear(64→1):")
    print(f"    Σw (all 64)   = {w.sum():.4f}")
    pos_w = w[w > 0]
    neg_w = w[w < 0]
    print(f"    Σw (positive) = {pos_w.sum():.4f}  ({len(pos_w)} neurons)")
    print(f"    Σw (negative) = {neg_w.sum():.4f}  ({len(neg_w)} neurons)")

    # Split contributions by sign of weight
    pos_contrib = contributions[w > 0].sum()
    neg_contrib = contributions[w < 0].sum()
    print(f"\n  Logit decomposition:")
    print(f"    Positive-weight neurons contribute: {pos_contrib:+.4f}")
    print(f"    Negative-weight neurons contribute: {neg_contrib:+.4f}")
    print(f"    Bias:                               {b:+.6f}")
    print(f"    Total logit:                        {logit.item():+.4f}")

    # Top contributors
    top_idx = np.argsort(np.abs(contributions))[::-1][:10]
    print(f"\n  Top 10 contributing neurons (|w·relu|):")
    print(f"    {'idx':>4}  {'w':>8}  {'relu':>8}  {'w·relu':>10}  {'cumsum':>10}")
    cumsum = 0
    for i in top_idx:
        cumsum += contributions[i]
        print(f"    {i:4d}  {w[i]:+8.4f}  {r[i]:8.4f}  {contributions[i]:+10.4f}  {cumsum:+10.4f}")

    # Neurons with relu=0 (killed by ReLU) — their negative weights don't contribute
    killed = np.where(r == 0)[0]
    killed_w = w[killed]
    neg_killed = killed_w[killed_w < 0]
    print(f"\n  Neurons killed by ReLU (r=0): {len(killed)}/64")
    print(f"    Of those, {len(neg_killed)} had negative weights (contribution suppressed to 0)")
    print(f"    Suppressed negative contribution: {neg_killed.sum():.4f}  ← would have subtracted this from logit")

    return logit.item(), score.item(), contributions


print("Loading models...")
gif_path = Path("fighting.gif")
vid_path = Path("data/test_video.mp4")

clip_g, _ = get_one_clip_features(str(gif_path), is_gif=True)
logit_g, score_g, contrib_g = decompose_logit(clip_g, "fighting.gif — clip 0")

if vid_path.exists():
    clip_v, _ = get_one_clip_features(str(vid_path), is_gif=False)
    logit_v, score_v, contrib_v = decompose_logit(clip_v, "test_video.mp4 — clip 0")

    print(f"\n{'='*60}")
    print("  ROOT CAUSE SUMMARY")
    print(f"{'='*60}")
    w = model.classifier[3].weight.detach().cpu().numpy().flatten()

    r_g = torch.nn.functional.relu(
        model.classifier[1](model.classifier[0](model.lstm(
            torch.FloatTensor(clip_g).unsqueeze(0).to(DEVICE)
        )[1][0][-1])).squeeze(0)
    ).detach().cpu().numpy()

    r_v = torch.nn.functional.relu(
        model.classifier[1](model.classifier[0](model.lstm(
            torch.FloatTensor(clip_v).unsqueeze(0).to(DEVICE)
        )[1][0][-1])).squeeze(0)
    ).detach().cpu().numpy()

    print(f"\n  relu activations comparison (first clip):")
    print(f"    {'':20}  {'fighting':>10}  {'normal':>10}  {'diff':>10}")
    print(f"    {'neurons active (>0)':20}  {(r_g>0).sum():>10}  {(r_v>0).sum():>10}")
    print(f"    {'relu max':20}  {r_g.max():>10.4f}  {r_v.max():>10.4f}  {r_g.max()-r_v.max():>+10.4f}")
    print(f"    {'relu mean':20}  {r_g.mean():>10.4f}  {r_v.mean():>10.4f}  {r_g.mean()-r_v.mean():>+10.4f}")
    print(f"    {'Σ(w·relu)':20}  {contrib_g.sum():>+10.4f}  {contrib_v.sum():>+10.4f}  {contrib_g.sum()-contrib_v.sum():>+10.4f}")

    print(f"""
  CONCLUSÃO:
  Os logits de ~+37 (fighting) e ~+39 (normal) são MUITO próximos e
  ambos estão > 13.8× acima do limiar de saturação da Sigmoid.

  O modelo NÃO está distinguindo os dois vídeos na prática.
  Isso indica que o treinamento com RankingLoss convergiu para uma
  solução degenerada: os pesos da LSTM aprenderam a produzir um
  estado oculto que, após o classificador, resulta em logit >>0
  para QUALQUER entrada de features realistas do ResNet50.

  O AUC=0.8881 no conjunto de validação foi obtido porque o modelo
  conseguia ordenar os scores (anomalia > normal) RELATIVAMENTE,
  mas os valores absolutos de ambas as classes ficaram saturados
  em score=1.0 — o que torna o threshold de 0.5 completamente inútil.

  Causa técnica provável:
  - O RankingLoss não penaliza scores absolutos altos — só exige
    score(anom) > score(normal) com margem 1.0.
  - Mas com Sigmoid como saída, a margem máxima possível é ~1.0
    (de 0 a 1), e o loss chega a 0 mesmo com ambas as saídas = 1.0,
    desde que a diferença seja ≥ 1.0 ANTES da Sigmoid (nos logits).
  - O modelo aprendeu a maximizar os logits para satisfazer a
    constraint nos logits (logit_anom - logit_normal >= 1.0),
    mas como os logits explodiram para +35~+41, ambos ficaram
    saturados na Sigmoid.
""")

print("Done.")
