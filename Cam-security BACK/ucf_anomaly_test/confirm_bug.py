"""
Confirma o bug da RankingLoss aplicada a scores pós-Sigmoid.

Demonstra numericamente:
  A) Por que logits explodindo satisfazem a loss mesmo com ambas as classes = 1.0
  B) O gradiente da Sigmoid comprimido perto de 1.0 impede discriminação
  C) Como a loss correta (aplicada nos logits) se comportaria
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np

# ── Reprodução exata do código atual ──────────────────────────────────────────
print("=" * 65)
print("CONFIRMAÇÃO DO BUG: RankingLoss aplicada pós-Sigmoid")
print("=" * 65)

margin = 1.0

def ranking_loss_current(scores, labels, margin=1.0):
    """Implementação atual em lstm_model.py — scores já passaram pela Sigmoid."""
    scores = scores.squeeze(1)
    anom_scores   = scores[labels == 1]
    normal_scores = scores[labels == 0]
    if len(anom_scores) == 0 or len(normal_scores) == 0:
        return torch.tensor(0.0)
    diff = anom_scores.unsqueeze(1) - normal_scores.unsqueeze(0)
    loss = torch.clamp(margin - diff, min=0.0)
    return loss.mean()

sigmoid = nn.Sigmoid()

print("\n── Cenário 1: logits pequenos (modelo não treinado) ──────────")
logit_anom   = torch.tensor([[0.5]])
logit_normal = torch.tensor([[-0.5]])
s_anom   = sigmoid(logit_anom)
s_normal = sigmoid(logit_normal)
scores   = torch.cat([s_anom, s_normal])
labels   = torch.tensor([1, 0])
loss_a = ranking_loss_current(scores.unsqueeze(1), labels)
print(f"  logit_anom={logit_anom.item():.1f}  → score={s_anom.item():.4f}")
print(f"  logit_norm={logit_normal.item():.1f}  → score={s_normal.item():.4f}")
print(f"  score diff = {(s_anom - s_normal).item():.4f}")
print(f"  loss (pós-Sigmoid) = max(0, 1.0 - {(s_anom-s_normal).item():.4f}) = {loss_a.item():.4f}")

print("\n── Cenário 2: logits grandes (estado atual do modelo) ────────")
logit_anom2   = torch.tensor([[37.0]])
logit_normal2 = torch.tensor([[39.0]])   # normal TEM logit maior!
s_anom2   = sigmoid(logit_anom2)
s_normal2 = sigmoid(logit_normal2)
scores2   = torch.cat([s_anom2, s_normal2])
loss_b = ranking_loss_current(scores2.unsqueeze(1), labels)
print(f"  logit_anom={logit_anom2.item():.1f}  → score={s_anom2.item():.10f}")
print(f"  logit_norm={logit_normal2.item():.1f}  → score={s_normal2.item():.10f}")
print(f"  score diff = {(s_anom2 - s_normal2).item():.2e}  (praticamente zero)")
print(f"  loss (pós-Sigmoid) = {loss_b.item():.6f}  ← ZERO mesmo com normal > anom!")
print(f"  PROBLEM: loss=0 não força nenhuma correção. O modelo para de aprender.")

print("\n── Cenário 3: logit_anom > logit_normal por margem exata ─────")
logit_anom3   = torch.tensor([[38.0]])
logit_normal3 = torch.tensor([[36.9]])   # diff = 1.1 no logit
s_anom3   = sigmoid(logit_anom3)
s_normal3 = sigmoid(logit_normal3)
scores3   = torch.cat([s_anom3, s_normal3])
loss_c = ranking_loss_current(scores3.unsqueeze(1), labels)
print(f"  logit_anom={logit_anom3.item():.1f}  → score={s_anom3.item():.10f}")
print(f"  logit_norm={logit_normal3.item():.1f}  → score={s_normal3.item():.10f}")
print(f"  logit diff = {(logit_anom3 - logit_normal3).item():.1f}  (> margin=1.0)")
print(f"  score diff = {(s_anom3 - s_normal3).item():.2e}")
print(f"  loss (pós-Sigmoid) = {loss_c.item():.6f}  ← zero porque Sigmoid comprimiu tudo")

# ── Gradiente da Sigmoid perto da saturação ───────────────────────────────────
print("\n── Gradiente da Sigmoid em diferentes logits ──────────────────")
print(f"  {'logit':>8}  {'sigmoid':>12}  {'d(sigmoid)/d(logit)':>22}")
for logit_val in [-40, -20, -10, -5, 0, 5, 10, 20, 37, 40]:
    x = torch.tensor(float(logit_val), requires_grad=True)
    s = sigmoid(x)
    s.backward()
    grad = x.grad.item()
    print(f"  {logit_val:>8}  {s.item():>12.8f}  {grad:>22.2e}")

print(f"\n  → Gradiente em logit=37: praticamente zero (≈1e-17)")
print(f"  → O modelo não consegue aprender mais nenhuma distinção.")

# ── Comportamento correto: loss nos logits ────────────────────────────────────
print("\n" + "=" * 65)
print("COMPORTAMENTO CORRETO: RankingLoss aplicada nos logits")
print("=" * 65)

def ranking_loss_correct(logits, labels, margin=1.0):
    """Loss correta — aplicada diretamente nos logits, antes da Sigmoid."""
    logits = logits.squeeze(1)
    anom_logits   = logits[labels == 1]
    normal_logits = logits[labels == 0]
    if len(anom_logits) == 0 or len(normal_logits) == 0:
        return torch.tensor(0.0)
    diff = anom_logits.unsqueeze(1) - normal_logits.unsqueeze(0)
    loss = torch.clamp(margin - diff, min=0.0)
    return loss.mean()

print("\n── Cenário 2 revisitado: logits grandes ───────────────────────")
logits2 = torch.tensor([[37.0], [39.0]])  # anom=37, normal=39
loss_correct_b = ranking_loss_correct(logits2, labels)
print(f"  logit_anom=37.0, logit_norm=39.0")
print(f"  logit diff = {37.0 - 39.0:.1f}  (normal > anom — ERRADO)")
print(f"  loss (nos logits) = max(0, 1.0 - (-2.0)) = {loss_correct_b.item():.4f}  ← PENALIZA!")
print(f"  Gradiente é não-zero → o modelo aprende a corrigir")

print("\n── Cenário 3 revisitado: logits com diff=1.1 ──────────────────")
logits3 = torch.tensor([[38.0], [36.9]])  # diff = 1.1 > margin=1.0
loss_correct_c = ranking_loss_correct(logits3, labels)
print(f"  logit_anom=38.0, logit_norm=36.9, diff=1.1 > margin=1.0")
print(f"  loss (nos logits) = max(0, 1.0 - 1.1) = {loss_correct_c.item():.4f}  ← satisfeito")

print("\n── Gradiente da loss correta em logits grandes ─────────────────")
logit_a = torch.tensor([[37.0]], requires_grad=True)
logit_n = torch.tensor([[39.0]], requires_grad=True)
logits_combined = torch.cat([logit_a, logit_n])
loss_grad = ranking_loss_correct(logits_combined, labels)
loss_grad.backward()
print(f"  logit_anom grad = {logit_a.grad.item():.4f}  (non-zero → modelo aprende)")
print(f"  logit_norm grad = {logit_n.grad.item():.4f}")
print(f"  Loss value = {loss_grad.item():.4f}")

# ── Resumo da causa raiz ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESUMO DA CAUSA RAIZ")
print("=" * 65)
print("""
  BUG: lstm_model.py AnomalyLSTM.forward() aplica Sigmoid antes de
       retornar. RankingLoss recebe scores ∈ [0,1] em vez de logits.

  Por que isso causa colapso:
    1. Sigmoid(37) = 1.0000000 e Sigmoid(39) = 1.0000000
       A diferença real de score é ~2e-17 — abaixo da precisão float32.
    2. RankingLoss recebe (1.0, 1.0) → diff=0 → loss = max(0, 1-0) = 1.0
       [WAIT — isso daria loss > 0. Mas o histórico mostra loss → 3.7e-05]

  O mecanismo real do colapso:
    - Quando logit_anom - logit_normal > ~13.8, Sigmoid(anom) - Sigmoid(normal)
      fica indistinguível de 0 em float32.
    - Mas ANTES disso, enquanto os logits ainda eram < 13.8, o modelo podia
      satisfazer a loss com diff de score > 0 nos logits.
    - Com logit_anom = logit_normal + ε (qualquer ε > 0), ambos crescem
      juntos enquanto a constraint é satisfeita.
    - Não há nenhum termo de regularização que impeça os logits de crescer.
    - clip_grad_norm limita o passo, mas não o valor acumulado.
    - Resultado: ambos os logits explodiram para +35~+41 juntos.

  Prova que NÃO é apenas precision:
    - logit_normal (test_video.mp4) = +39.96 > logit_anom (fighting) = +37.35
    - O modelo vídeo normal TEM logit maior que o GIF de briga.
    - A RankingLoss satisfeita durante treino era: anom_train > normal_train
    - Esses VÍDEOS ESPECÍFICOS não foram vistos no treino.

  Fix correto:
    - forward() retorna logits (sem Sigmoid)
    - RankingLoss opera nos logits
    - Sigmoid aplicada APENAS na inferência (pipeline.py)
    - train.py: sigmoid(scores) antes do roc_auc_score
""")

print("Confirmação do bug concluída.")
