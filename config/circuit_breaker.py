import datetime
import threading

from config.configs_params import BACKEND_HEALTH
from config.remediation import _auto_restart_vllm
from configs_params import (_lock, CIRCUIT_OPEN, _alert
                            )

CB_THRESHOLD = 3
CB_RESET_SECONDS = 30

_cb_failures = 0
_cb_open = False
_cb_reset_after: datetime | None = None

def cb_is_open()-> bool:
    global _cb_open, _cb_failures, _cb_reset_after
    with _lock:
        if _cb_open and _cb_reset_after and datetime.now() > _cb_reset_after:
            _cb_open = False
            _cb_failures = 0
            CIRCUIT_OPEN.set(0)
            _alert("INFO", "circuit_closed","Circuit breaker CLOSED- retrying vLLM backend")
    return _cb_open

def cb_record_success():
    global _cb_failures, _cb_open
    with _lock:
        _cb_failures = 0
        _cb_open = False
        BACKEND_HEALTH.set(1)
        CIRCUIT_OPEN.set(0)

def cb_record_failure(reason: str):
    global _cb_failures, _cb_open, _cb_reset_after
    with _lock:
        _cb_failures +=1
        BACKEND_HEALTH.set(0)
        if not _cb_open and _cb_failures >= CB_THRESHOLD:
            _cb_open = True
            _cb_reset_after = datetime.now() + datetime.timedelta(seconds=CB_RESET_SECONDS)
            CIRCUIT_OPEN.set(1)
            just_opened = True
            _alert("CRITICAL", "circuit_open",
                   f"Circuit breaker OPEN after {_cb_failures} failures ({reason}). "
                   f"Routing to local fallback — auto-restarting vLLM")
    if just_opened:
        threading.Thread(target= _auto_restart_vllm, daemon=True).start()