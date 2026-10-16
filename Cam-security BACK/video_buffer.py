import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger("camsecurity.video_buffer")


@dataclass
class BufferedFrame:
    """Frame com metadados para gravação de vídeo."""
    frame: np.ndarray
    seq: int
    timestamp: float


RecordingDoneCallback = Callable[[List[BufferedFrame], List[BufferedFrame], dict], None]


class VideoBuffer:
    """
    Buffer circular thread-safe para frames de vídeo.

    Armazena uma janela deslizante de frames para permitir
    salvar vídeo do período anterior a um evento de anomalia.
    """

    def __init__(self, buffer_seconds: float = 5.0, fps: float = 30.0):
        self._seconds = buffer_seconds
        self._fps = fps
        max_frames = int(buffer_seconds * fps * 1.2)  # 20% de margem
        self._buffer: deque[BufferedFrame] = deque(maxlen=max_frames)

        import threading
        self._lock = threading.Lock()

        # Estado da gravação
        self._recording = False
        self._rec_frames: List[BufferedFrame] = []
        self._rec_pre: List[BufferedFrame] = []
        self._rec_target = 0
        self._rec_meta: dict = {}
        self._on_done: Optional[RecordingDoneCallback] = None

        logger.info("VideoBuffer: %.1fs @ %.1f FPS (máx %d frames)", buffer_seconds, fps, max_frames)

    def push_frame(self, frame: np.ndarray, seq: int, timestamp: float) -> None:
        """Adiciona frame ao buffer circular. Se gravando, acumula frames pós-evento."""
        done_cb = None
        pre, post, meta = [], [], {}

        with self._lock:
            bf = BufferedFrame(frame=frame.copy(), seq=seq, timestamp=timestamp)
            self._buffer.append(bf)

            if self._recording:
                self._rec_frames.append(bf)
                if len(self._rec_frames) >= self._rec_target:
                    self._recording = False
                    pre, post, meta = list(self._rec_pre), list(self._rec_frames), dict(self._rec_meta)
                    done_cb = self._on_done
                    self._rec_frames, self._rec_pre, self._rec_meta, self._on_done = [], [], {}, None
                    logger.debug("Gravação concluída: %d pre + %d post frames", len(pre), len(post))

        if done_cb:
            try:
                done_cb(pre, post, meta)
            except Exception:
                logger.exception("Erro no callback de gravação")

    def start_recording(self, duration_seconds: float, event_timestamp: float,
                        pre_seconds: float, on_done: RecordingDoneCallback,
                        metadata: Optional[dict] = None) -> None:
        """Inicia gravação pós-evento e registra callback de conclusão."""
        with self._lock:
            if self._recording:
                logger.warning("Gravação já em andamento — nova solicitação ignorada")
                return

            cutoff = event_timestamp - pre_seconds
            pre = [bf for bf in self._buffer if cutoff <= bf.timestamp <= event_timestamp]
            target = max(1, int(duration_seconds * self._fps))

            self._recording = True
            self._rec_frames = []
            self._rec_pre = pre
            self._rec_target = target
            self._rec_meta = metadata or {}
            self._on_done = on_done

            logger.info("Gravação iniciada: %.1fs pós-evento (%d frames alvo), %d frames pré-evento",
                        duration_seconds, target, len(pre))

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def cancel_recording(self) -> None:
        with self._lock:
            if self._recording:
                self._recording = False
                self._rec_frames, self._rec_pre = [], []
                self._rec_target, self._rec_meta, self._on_done = 0, {}, None
                logger.info("Gravação cancelada")

    def resize_buffer(self, fps: float) -> None:
        """Redimensiona o buffer para um novo FPS real da fonte."""
        with self._lock:
            new_max = int(self._seconds * fps * 1.2)
            if new_max == self._buffer.maxlen:
                return
            from collections import deque as _deque
            old_max = self._buffer.maxlen
            self._buffer = _deque(list(self._buffer), maxlen=new_max)
            self._fps = fps
            logger.info("Buffer redimensionado: %d → %d frames (%.1f FPS)", old_max, new_max, fps)

    def get_buffer_duration(self) -> float:
        with self._lock:
            return (self._buffer[-1].timestamp - self._buffer[0].timestamp) if len(self._buffer) >= 2 else 0.0

    def get_buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._recording = False
            self._rec_frames, self._rec_pre = [], []
            self._rec_target, self._rec_meta, self._on_done = 0, {}, None
            logger.info("Buffer limpo")


def save_video(frames: List[BufferedFrame], output_path: str,
               fps: float = 30.0, codec: str = "XVID") -> bool:
    """Salva lista de frames em arquivo de vídeo. Retorna True em sucesso."""
    import cv2
    if not frames:
        logger.warning("save_video: nenhum frame fornecido")
        return False
    try:
        h, w = frames[0].frame.shape[:2]
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if not writer.isOpened():
            logger.error("save_video: falha ao abrir VideoWriter para %s", output_path)
            return False
        for bf in frames:
            writer.write(bf.frame)
        writer.release()
        logger.info("Vídeo salvo: %s  %d frames  %.1fs", output_path, len(frames), len(frames) / fps)
        return True
    except Exception:
        logger.exception("save_video: erro ao gravar %s", output_path)
        return False
