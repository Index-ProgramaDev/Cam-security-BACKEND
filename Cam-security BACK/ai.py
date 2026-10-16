import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from ucf_adapter import UCFStreamAdapter

logger = logging.getLogger("camsecurity.ai")
_ROOT = Path(__file__).parent


@dataclass
class YOLODetection:
    """Resultado de uma detecção YOLO por frame."""
    class_name: str
    confidence: float
    bbox: tuple        # (x1, y1, x2, y2)
    track_id: int = None


@dataclass
class AnomalyResult:
    """Resultado completo de um frame processado pela IA."""
    frame_seq: int
    timestamp: float
    yolo_detections: List[YOLODetection] = field(default_factory=list)
    ucf_score: Optional[float] = None
    new_ucf_score: bool = False
    is_anomalous: bool = False
    annotated_frame: Optional[np.ndarray] = None
    yolo_ms: float = 0.0
    ucf_ms: float = 0.0


class BBoxTracker:
    """Mantém IDs de suspeitos e prolonga alerta ativo."""

    def __init__(self, hold_alert_sec: float = 2.0):
        self._suspects: set = set()
        self._last_alert: float = 0.0
        self._hold = hold_alert_sec

    def mark_suspects(self, track_ids) -> None:
        novos = {tid for tid in track_ids if tid is not None}
        if novos:
            self._suspects.update(novos)
            self._last_alert = time.perf_counter()
            logger.info("Suspeitos marcados: %s", sorted(novos))

    def update(self, detections, is_anomalous: bool):
        now = time.perf_counter()
        if is_anomalous:
            self._last_alert = now
        visivel = any(d.track_id in self._suspects for d in detections if d.track_id is not None)
        is_anomalous_out = is_anomalous or (now - self._last_alert) < self._hold or visivel
        return detections, is_anomalous_out

    def get_suspects(self) -> set:
        return set(self._suspects)

    def reset(self) -> None:
        self._suspects.clear()
        self._last_alert = 0.0


class AnomalyProcessor:
    """Carrega YOLO e UCF uma vez e processa frames de forma síncrona."""

    def __init__(self, config: dict):
        ai = config.get("ai", {})
        ucf = config.get("ucf", {})

        self._yolo_path    = str(ai.get("yolo_model_path", _ROOT / "yolo11n.pt"))
        self._yolo_conf    = float(ai.get("yolo_conf", 0.25))
        self._yolo_classes = ai.get("yolo_classes", None)
        self._device       = str(ai.get("device", "cpu"))
        self._draw         = bool(ai.get("draw_overlays", True))
        self._imgsz        = int(ai.get("imgsz", 640))
        self._iou          = float(ai.get("iou", 0.45))   # NMS IoU — menor = menos caixas duplicadas
        self._tracking     = bool(ai.get("enable_tracking", False))
        self._logged_dims  = False   # loga dimensões do frame apenas uma vez

        self._ucf_ckpt   = str(ucf.get("checkpoint", _ROOT / "ucf_anomaly_test" / "checkpoints" / "best_model.pt"))
        self._ucf_thresh = float(ucf.get("threshold", 0.5))
        self._ucf_clen   = int(ucf.get("clip_length", 16))
        self._ucf_stride = int(ucf.get("clip_stride", 8))

        self._tracker = BBoxTracker(hold_alert_sec=float(ai.get("hold_alert_sec", 2.0)))
        self._yolo = None
        self._ucf = None
        self._ready = False

    def start(self) -> None:
        """Carrega os modelos YOLO e UCF."""
        logger.info("Carregando YOLO de %s ...", self._yolo_path)
        self._yolo = YOLO(self._yolo_path)
        logger.info("Carregando UCF (ResNet50 + LSTM) ...")
        self._ucf = UCFStreamAdapter(
            checkpoint_path=self._ucf_ckpt,
            device=self._device,
            clip_length=self._ucf_clen,
            clip_stride=self._ucf_stride,
            threshold=self._ucf_thresh,
        )
        self._ready = True
        logger.info("Modelos prontos.")

    def process_frame(self, frame: np.ndarray, seq: int, ts: float) -> AnomalyResult:
        """Executa YOLO + UCF em um frame e retorna o resultado."""
        if not self._ready:
            raise RuntimeError("Chame start() antes de process_frame().")

        t0 = time.perf_counter()
        detections = self._run_yolo(frame)
        yolo_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        new_score = self._ucf.push_frame(frame)
        ucf_ms = (time.perf_counter() - t0) * 1000

        new_ucf = new_score is not None
        ucf_score = new_score if new_ucf else (self._ucf.get_last_score() or None)
        is_anomalous = ucf_score is not None and ucf_score >= self._ucf_thresh

        detections, is_anomalous = self._tracker.update(detections, is_anomalous)

        if yolo_ms > 150:
            logger.warning("YOLO lento | frame=%d  %.1fms", seq, yolo_ms)
        if new_ucf:
            logger.info("UCF | frame=%d  score=%.4f  pessoas=%d  %s",
                        seq, ucf_score, len(detections),
                        "ANOMALIA" if is_anomalous else "normal")

        annotated = self._draw_overlays(frame, detections, ucf_score, is_anomalous, yolo_ms, ucf_ms) if self._draw else frame

        return AnomalyResult(
            frame_seq=seq, timestamp=ts,
            yolo_detections=detections, ucf_score=ucf_score,
            new_ucf_score=new_ucf, is_anomalous=is_anomalous,
            annotated_frame=annotated, yolo_ms=yolo_ms, ucf_ms=ucf_ms,
        )

    def mark_suspects(self, track_ids) -> None:
        self._tracker.mark_suspects(track_ids)

    def stop(self) -> None:
        if self._ucf:
            self._ucf.reset()
        self._tracker.reset()
        self._ready = False
        logger.info("AnomalyProcessor encerrado.")

    def _run_yolo(self, frame: np.ndarray) -> List[YOLODetection]:
        """Roda YOLO em um frame e retorna lista de detecções."""
        # Loga resolução do frame apenas na primeira chamada para diagnóstico
        if not self._logged_dims:
            h, w = frame.shape[:2]
            logger.info("Frame recebido pelo YOLO: %dx%d  imgsz=%d  conf=%.2f  iou=%.2f",
                        w, h, self._imgsz, self._yolo_conf, self._iou)
            self._logged_dims = True

        kwargs = dict(
            source=frame, conf=self._yolo_conf, classes=self._yolo_classes,
            device=self._device, imgsz=self._imgsz, iou=self._iou, verbose=False,
        )
        results = self._yolo.track(persist=True, **kwargs) if self._tracking else self._yolo.predict(**kwargs)

        detections = []
        for box in results[0].boxes:
            cid = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            tid = int(box.id[0]) if self._tracking and box.id is not None else None
            detections.append(YOLODetection(self._yolo.names[cid], float(box.conf[0]), (x1, y1, x2, y2), tid))
        return detections

    def _draw_overlays(self, frame: np.ndarray, detections, ucf_score, is_anomalous, yolo_ms=0.0, ucf_ms=0.0) -> np.ndarray:
        """Desenha bounding boxes e painel de status no frame."""
        out = frame.copy()
        H, W = out.shape[:2]
        suspects = self._tracker.get_suspects()

        for d in detections:
            x1, y1, x2, y2 = d.bbox
            suspeito = d.track_id is not None and d.track_id in suspects
            color = (0, 0, 255) if (suspeito or is_anomalous) else (0, 180, 255)
            thick = 3 if suspeito or is_anomalous else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            label = f"{'[!] ' if suspeito else ''}{d.class_name} {d.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(out, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Painel superior
        cv2.rectangle(out, (0, 0), (W, 50), (20, 20, 20), -1)
        if ucf_score is None:
            status, score_txt, cor = "UCF: aquecendo...", "", (100, 100, 100)
        elif is_anomalous:
            status, score_txt, cor = "ANOMALIA DETECTADA", f"score={ucf_score:.3f}", (0, 0, 220)
        else:
            status, score_txt, cor = "Normal", f"score={ucf_score:.3f}", (0, 180, 0)

        cv2.rectangle(out, (0, 0), (W, 50), cor, 3)
        cv2.putText(out, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if score_txt:
            cv2.putText(out, score_txt, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        timing = f"YOLO:{yolo_ms:.0f}ms  UCF:{ucf_ms:.0f}ms"
        (ttw, _), _ = cv2.getTextSize(timing, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(out, timing, (W - ttw - 6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        return out
