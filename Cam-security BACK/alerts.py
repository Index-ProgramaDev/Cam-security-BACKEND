import logging
import threading
import time
import winsound

logger = logging.getLogger("camsecurity.alerts")


class AlertaSonoro:
    """Alerta sonoro com cooldown para evitar disparos repetidos."""

    def __init__(self, cooldown=5):
        self.cooldown = cooldown
        self._ultimo = 0.0
        self._lock = threading.Lock()

    def disparar(self, motivo="Anomalia detectada", score=None) -> bool:
        """Dispara alerta sonoro se o cooldown já passou. Retorna True se disparou."""
        with self._lock:
            agora = time.time()
            if agora - self._ultimo < self.cooldown:
                return False
            self._ultimo = agora

        detalhe = f" | Score: {score:.2f}" if score is not None else ""
        logger.warning("ALERTA: %s%s", motivo, detalhe)
        threading.Thread(target=self._tocar, daemon=True).start()
        return True

    @staticmethod
    def _tocar():
        try:
            winsound.MessageBeep(winsound.MB_ICONHAND)
        except Exception:
            pass
