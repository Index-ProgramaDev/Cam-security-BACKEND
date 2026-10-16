import cv2
import os
import time
import yaml
import logging
import threading

logger = logging.getLogger("camsecurity.capture")


class CameraCapture:
    """Captura frames de câmera ou arquivo de vídeo em thread dedicada."""

    def __init__(self, config_path="config.yaml", shared_state=None, video_buffer=None):
        self.width, self.height, self.fps = 640, 480, 30
        self.camera_id, self.video_path = 0, ""
        self.config_path = config_path
        self.cap = None
        self.source = None
        self.is_video = False
        self.running = False
        self.frame = None
        self.frame_seq = 0
        self.frame_captured_at = 0.0
        self.lock = threading.Lock()
        self.thread = None
        self.shared_state = shared_state
        self.video_buffer = video_buffer
        self._load_config()

    def _load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                for key in ("camera_id", "video_path", "width", "height", "fps"):
                    if key in cfg:
                        setattr(self, key, cfg[key])
        except Exception as e:
            logger.error("Erro ao carregar config: %s", e)

    def start(self):
        if self.running:
            return

        self.is_video = bool(self.video_path and os.path.exists(self.video_path)) or (
            isinstance(self.camera_id, str) and os.path.exists(self.camera_id)
        )
        self.source = (self.video_path if self.video_path and os.path.exists(self.video_path)
                       else self.camera_id)

        self.cap = cv2.VideoCapture(self.source)

        # Tenta outras câmeras se a configurada falhar
        if not self.cap.isOpened() and not isinstance(self.source, str):
            for cid in range(4):
                if cid == self.camera_id:
                    continue
                self.cap.release()
                self.cap = cv2.VideoCapture(cid)
                if self.cap.isOpened():
                    self.camera_id = self.source = cid
                    break

        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a fonte: {self.source}")

        if self.is_video:
            real_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if 1 <= real_fps <= 120:
                self.fps = real_fps
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.frame_interval = 1 / max(self.fps, 1)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info("Fonte iniciada: %s", self.source)

    @property
    def real_fps(self) -> float:
        return self.fps

    def _loop(self):
        next_frame = time.perf_counter()
        while self.running:
            if self.is_video:
                delay = next_frame - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

            ret, frame = self.cap.read() if self.cap else (False, None)
            if not ret or frame is None:
                if self.is_video:
                    # Reinicia o vídeo ao terminar
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    next_frame = time.perf_counter()
                else:
                    time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (self.width, self.height))
            now = time.perf_counter()

            with self.lock:
                self.frame = frame
                self.frame_seq += 1
                self.frame_captured_at = now

            if self.shared_state:
                self.shared_state.update_frame(frame, self.frame_seq, now)
            if self.video_buffer:
                self.video_buffer.push_frame(frame, self.frame_seq, now)

            if self.is_video:
                next_frame += self.frame_interval
                if next_frame < time.perf_counter():
                    next_frame = time.perf_counter()

    def get_frame(self, return_meta=False):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            seq, ts = self.frame_seq, self.frame_captured_at
        return (frame, seq, ts) if return_meta else frame

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Câmera encerrada.")