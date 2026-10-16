import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional

from alerts import AlertaSonoro

logger = logging.getLogger("camsecurity.events")


class EventKind(Enum):
    ANOMALY_CONFIRMED = auto()
    ANOMALY_CLEARED   = auto()


@dataclass
class Event:
    kind:            EventKind
    frame_seq:       int
    timestamp:       float
    ucf_score:       float
    yolo_summary:    str
    frame:           object = None
    yolo_detections: object = None


class EventManager:
    """Confirmação de anomalia por janela deslizante com alerta sonoro."""

    def __init__(self, config: dict):
        ev  = config.get("events", {})
        ucf = config.get("ucf", {})

        self._cooldown  = float(ev.get("alert_cooldown_sec", 5))
        self._window    = int(ev.get("confirm_window", 5))
        self._min_hits  = int(ev.get("confirm_min_hits", 3))
        self._threshold = float(ucf.get("threshold", 0.5))

        self._history: deque = deque(maxlen=self._window)
        self._in_anomaly = False
        self._alert = AlertaSonoro(cooldown=self._cooldown)
        self._callbacks: List[Callable] = []

    def on_event(self, callback: Callable) -> None:
        """Registra callback chamado em cada evento confirmado/limpo."""
        self._callbacks.append(callback)

    def set_storage_manager(self, storage_manager) -> None:
        """Mantido por compatibilidade — não utilizado internamente."""
        pass

    def process(self, result) -> Optional[Event]:
        """Processa um AnomalyResult e retorna Event em transição de estado."""
        resumo = _resumir_deteccoes(result.yolo_detections)

        # Registra na janela apenas quando o LSTM gera um novo score
        if getattr(result, "new_ucf_score", False) and result.ucf_score is not None:
            anomalo = result.ucf_score >= self._threshold
            self._history.append(anomalo)
            if anomalo:
                logger.info("Anomalia detectada | frame=%d  score=%.4f  %s",
                            result.frame_seq, result.ucf_score, resumo)

        hits = sum(self._history)
        confirmado = len(self._history) == self._window and hits >= self._min_hits

        event = None

        if confirmado and not self._in_anomaly:
            self._in_anomaly = True
            event = Event(
                kind=EventKind.ANOMALY_CONFIRMED,
                frame_seq=result.frame_seq, timestamp=result.timestamp,
                ucf_score=result.ucf_score or 0.0, yolo_summary=resumo,
                frame=result.annotated_frame, yolo_detections=result.yolo_detections,
            )
            logger.warning("ANOMALIA CONFIRMADA | frame=%d  score=%.4f  hits=%d/%d  %s",
                           result.frame_seq, result.ucf_score or 0.0, hits, self._window, resumo)
            self._alert.disparar(motivo=f"Anomalia confirmada — {resumo}", score=result.ucf_score)
            self._dispatch(event)

        elif not confirmado and self._in_anomaly:
            self._in_anomaly = False
            event = Event(
                kind=EventKind.ANOMALY_CLEARED,
                frame_seq=result.frame_seq, timestamp=result.timestamp,
                ucf_score=result.ucf_score or 0.0, yolo_summary=resumo,
                frame=result.annotated_frame,
            )
            logger.info("ANOMALIA ENCERRADA | frame=%d  score=%.4f  %s",
                        result.frame_seq, result.ucf_score or 0.0, resumo)
            self._dispatch(event)

        return event

    @property
    def in_anomaly(self) -> bool:
        return self._in_anomaly

    def reset(self) -> None:
        self._history.clear()
        self._in_anomaly = False

    def _dispatch(self, event: Event) -> None:
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                logger.exception("Erro no callback de evento")


def _resumir_deteccoes(deteccoes) -> str:
    if not deteccoes:
        return "sem detecções"
    return ", ".join(f"{d.class_name}({d.confidence:.2f})" for d in deteccoes[:5])
