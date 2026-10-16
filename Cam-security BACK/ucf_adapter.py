import logging
import sys
import threading
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from collections import deque
from typing import Optional

# Adiciona o pacote ucf_anomaly_test ao path
_UCF_ROOT = Path(__file__).parent / "ucf_anomaly_test"
if str(_UCF_ROOT) not in sys.path:
    sys.path.insert(0, str(_UCF_ROOT))

# pyrefly: ignore [missing-import]
    from src.models.feature_extractor import ResNet50Extractor
# pyrefly: ignore [missing-import]
from src.models.lstm_model import AnomalyLSTM

logger = logging.getLogger("camsecurity.ucf")

# Pré-processamento padrão ImageNet (igual ao treinamento)
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_DEFAULT_CHECKPOINT = _UCF_ROOT / "checkpoints" / "best_model.pt"


class UCFStreamAdapter:
    """
    Adapter para inferência em stream ao vivo com ResNet50 + LSTM.

    Recebe frames BGR um a um via push_frame() e retorna score de anomalia
    a cada clip_stride frames (None nos demais).
    """

    def __init__(self, checkpoint_path=None, device="cpu",
                 clip_length=16, clip_stride=8, threshold=0.5, feature_batch_size=16):
        self.device        = device
        self.clip_length   = clip_length
        self.clip_stride   = clip_stride
        self.threshold     = threshold
        self.feature_batch = feature_batch_size

        ckpt_path = Path(checkpoint_path or _DEFAULT_CHECKPOINT)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint UCF não encontrado: {ckpt_path}")

        logger.info("Carregando ResNet50 extrator de features...")
        self._extractor = ResNet50Extractor().to(device)
        self._extractor.eval()

        logger.info("Carregando AnomalyLSTM de %s ...", ckpt_path.name)
        self._lstm = AnomalyLSTM().to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        self._lstm.load_state_dict(ckpt["model_state"])
        self._lstm.eval()
        logger.info("UCF pronto  (epoch=%d  val_auc=%.4f  threshold=%.2f)",
                    ckpt["epoch"], ckpt["val_auc"], threshold)

        self._buffer: deque = deque(maxlen=clip_length)
        self._cache: dict = {}
        self._frames_since_score = 0
        self._last_score = 0.0
        self._total_frames = 0

    def push_frame(self, frame_bgr: np.ndarray) -> Optional[float]:
        """
        Envia um frame BGR. Retorna score [0,1] quando um clip é pontuado,
        ou None nos demais frames.
        """
        pil = _bgr_para_pil(frame_bgr)
        self._buffer.append(pil)
        self._frames_since_score += 1
        self._total_frames += 1

        if len(self._buffer) == self.clip_length and self._frames_since_score >= self.clip_stride:
            score = self._pontuar_clip()
            self._frames_since_score = 0
            self._last_score = score
            if len(self._cache) > self.clip_length * 2:
                self._cache.clear()
            return score

        return None

    def get_last_score(self) -> float:
        return self._last_score

    def is_anomalous(self) -> bool:
        return self._last_score >= self.threshold

    def reset(self) -> None:
        self._buffer.clear()
        self._cache.clear()
        self._frames_since_score = 0
        self._last_score = 0.0
        self._total_frames = 0

    def _pontuar_clip(self) -> float:
        """Extrai features do buffer atual e roda o LSTM."""
        import time
        frames = list(self._buffer)

        t0 = time.perf_counter()
        features = self._extrair_features(frames)
        resnet_ms = (time.perf_counter() - t0) * 1000

        clip_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            score_tensor = self._lstm(clip_tensor)
        lstm_ms = (time.perf_counter() - t0) * 1000

        score = float(score_tensor.squeeze())
        logger.debug("Clip pontuado: score=%.4f  ResNet=%.1fms  LSTM=%.1fms  (frame %d)",
                     score, resnet_ms, lstm_ms, self._total_frames)
        return score

    def _extrair_features(self, frames: list) -> np.ndarray:
        """Extrai features ResNet50 em lotes com cache por frame."""
        all_feats = []
        for i in range(0, len(frames), self.feature_batch):
            batch = frames[i:i + self.feature_batch]
            feats = [None] * len(batch)
            sem_cache = []
            idx_sem_cache = []

            for j, pil in enumerate(batch):
                h = hash(pil.tobytes())
                if h in self._cache:
                    feats[j] = self._cache[h]
                else:
                    sem_cache.append(pil)
                    idx_sem_cache.append(j)
                    feats[j] = None

            if sem_cache:
                tensors = torch.stack([_TRANSFORM(f) for f in sem_cache]).to(self.device)
                with torch.no_grad():
                    novos = self._extractor(tensors).cpu().numpy()
                for k, (pil, feat) in enumerate(zip(sem_cache, novos)):
                    self._cache[hash(pil.tobytes())] = feat
                    feats[idx_sem_cache[k]] = feat

            all_feats.extend(feats)
        return np.array(all_feats)


def _bgr_para_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Converte frame BGR do OpenCV para imagem PIL RGB."""
    import cv2
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
