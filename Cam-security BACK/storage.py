import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("camsecurity.storage")


class StorageManager:
    """
    Persiste eventos de anomalia em disco.

    Estrutura de diretórios:
      storage/
        events.jsonl
        events/
          2024-01-15_14-32-07_0001/
            frame.jpg
            event.avi
    """

    def __init__(self, config: dict):
        st  = config.get("storage", {})
        vid = config.get("video", {})

        root = Path(st.get("root", "storage"))
        self._events_dir  = root / "events"
        self._log_path    = root / "events.jsonl"
        self._save_frames = bool(st.get("save_frames", True))
        self._jpeg_q      = int(st.get("jpeg_quality", 85))
        self._max_events  = int(st.get("max_events", 500))
        self._event_count = 0

        self._pre_sec   = float(vid.get("pre_event_seconds", 5))
        self._post_sec  = float(vid.get("post_event_seconds", 15))
        self._vid_fps   = float(vid.get("fps", 30))
        self._vid_codec = str(vid.get("codec", "XVID"))

        self._video_buffer = None
        self._log_lock = threading.Lock()

        self._events_dir.mkdir(parents=True, exist_ok=True)
        if self._log_path.exists():
            with open(self._log_path, encoding="utf-8") as f:
                self._event_count = sum(1 for _ in f)

        logger.info("StorageManager pronto — raiz=%s  eventos=%d  vídeo=%ds pre + %ds post",
                    root, self._event_count, int(self._pre_sec), int(self._post_sec))

    def set_video_buffer(self, video_buffer) -> None:
        self._video_buffer = video_buffer
        logger.info("VideoBuffer conectado ao StorageManager")

    def set_capture_fps(self, fps: float) -> None:
        if fps > 0 and fps != self._vid_fps:
            logger.info("FPS do vídeo: %.1f → %.1f", self._vid_fps, fps)
            self._vid_fps = fps

    def save_event(self, event) -> Path:
        """Salva frame e inicia gravação de vídeo para o evento confirmado."""
        from events import EventKind

        with self._log_lock:
            seq_str   = f"{self._event_count + 1:04d}"
            ts_str    = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
            event_dir = self._events_dir / f"{ts_str}_{seq_str}"
            event_dir.mkdir(parents=True, exist_ok=True)

            # Salva frame representativo
            frame_rel = None
            if (self._save_frames and event.frame is not None
                    and isinstance(event.frame, np.ndarray) and event.frame.size > 0):
                fp = event_dir / "frame.jpg"
                cv2.imwrite(str(fp), event.frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_q])
                frame_rel = str(fp.relative_to(self._events_dir.parent))

            # Inicia gravação de vídeo via VideoBuffer
            video_path = event_dir / "event.avi"
            video_rel = None
            if self._video_buffer and event.kind == EventKind.ANOMALY_CONFIRMED:
                self._video_buffer.start_recording(
                    duration_seconds=self._post_sec,
                    event_timestamp=event.timestamp,
                    pre_seconds=self._pre_sec,
                    on_done=self._on_recording_done,
                    metadata={"event_id": seq_str, "video_path": video_path},
                )
                video_rel = str(video_path.relative_to(self._events_dir.parent))

            # Registra no JSONL
            record = {
                "event_id": seq_str, "kind": event.kind.name,
                "timestamp": time.time(), "frame_seq": event.frame_seq,
                "ucf_score": round(event.ucf_score, 4), "detections": event.yolo_summary,
                "frame_path": frame_rel, "video_path": None,
            }
            self._rotate_if_needed()
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            self._event_count += 1
            logger.info("Evento salvo: %s  tipo=%s  score=%.3f", seq_str, event.kind.name, event.ucf_score)

        return event_dir

    def load_events(self) -> list:
        """Carrega todos os eventos do JSONL como lista de dicts."""
        if not self._log_path.exists():
            return []
        records = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Linha malformada ignorada no events.jsonl")
        return records

    def _on_recording_done(self, pre: list, post: list, meta: dict) -> None:
        """Callback do VideoBuffer — escreve o vídeo em thread separada."""
        t = threading.Thread(
            target=self._write_video,
            args=(pre, post, meta.get("video_path"), meta.get("event_id")),
            daemon=True, name=f"video-{meta.get('event_id')}",
        )
        t.start()

    def _write_video(self, pre: list, post: list, output_path: Path, event_id: Optional[str]) -> None:
        """Grava frames pré + pós em arquivo de vídeo e atualiza o JSONL."""
        frames = pre + post
        if not frames:
            logger.warning("Nenhum frame para gravar no evento %s", event_id)
            return
        try:
            h, w = frames[0].frame.shape[:2]
            writer = cv2.VideoWriter(str(output_path),
                                     cv2.VideoWriter_fourcc(*self._vid_codec),
                                     self._vid_fps, (w, h))
            if not writer.isOpened():
                logger.error("VideoWriter falhou para %s", output_path)
                return
            for bf in frames:
                writer.write(bf.frame)
            writer.release()

            duration = len(frames) / self._vid_fps
            video_rel = str(output_path.relative_to(self._events_dir.parent))
            logger.info("Vídeo salvo: %s  %d frames (pre=%d post=%d)  %.1fs",
                        output_path.name, len(frames), len(pre), len(post), duration)
            if event_id:
                self._patch_jsonl(event_id, video_rel)
        except Exception:
            logger.exception("Erro ao gravar vídeo %s", output_path)

    def _patch_jsonl(self, event_id: str, video_rel: str) -> None:
        """Atualiza o campo video_path do evento no JSONL."""
        with self._log_lock:
            try:
                lines = self._log_path.read_text(encoding="utf-8").splitlines(keepends=True)
                for i, line in enumerate(lines):
                    try:
                        rec = json.loads(line)
                        if rec.get("event_id") == event_id:
                            rec["video_path"] = video_rel
                            lines[i] = json.dumps(rec) + "\n"
                            break
                    except json.JSONDecodeError:
                        continue
                self._log_path.write_text("".join(lines), encoding="utf-8")
                logger.info("JSONL atualizado: evento %s → %s", event_id, video_rel)
            except Exception:
                logger.exception("Falha ao atualizar JSONL para evento %s", event_id)

    def _rotate_if_needed(self) -> None:
        """Rotaciona o JSONL quando ultrapassa o limite de eventos."""
        if self._event_count >= self._max_events:
            bak = self._log_path.with_suffix(".jsonl.bak")
            self._log_path.rename(bak)
            self._event_count = 0
            logger.info("events.jsonl rotacionado → %s", bak)
