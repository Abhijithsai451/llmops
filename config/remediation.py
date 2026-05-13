# ─── Auto-Remediation State ───────────────────────────────────────────────────
import datetime
import subprocess
import threading
from collections import deque
import requests

from config.alerting import _alert
from config.circuit_breaker import cb_record_success
from config.configs_params import REMEDIATION_COUNT, TOKEN_CAP_GAUGE, THROTTLE_GAUGE
from config.vllm_config import VLLM_CMD, VLLM_HEALTH

remediation_log: deque = deque(maxlen=50)
_auto_token_cap = 100
_throttle_delay = 0.0
_vllm_process   = None


def _remediate(action: str, reason: str):
    remediation_log.appendleft({
        "action": action,
        "reason": reason,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    REMEDIATION_COUNT.labels(action=action[:40]).inc()
    _alert("INFO", f"rem_{action[:25]}", f"AUTO-FIX: {action} — {reason}")

def _auto_adjust_token_cap(w_mean:float | None):
    global _auto_token_cap
    if w_mean is None:
        return
    if w_mean >= 3.5 and _auto_token_cap > 20:
        _auto_token_cap = 20
        TOKEN_CAP_GAUGE.set(20)
        _remediate("Token cap -> 20" , f"avg latency {w_mean:.2f}s > 3.5s — reducing output load")
    elif w_mean > 2.0 and _auto_token_cap> 50:
        _auto_token_cap = 50
        TOKEN_CAP_GAUGE.set(50)
        _remediate("Token cap → 50", f"avg latency {w_mean:.2f}s > 2s — throttling output size")
    elif w_mean < 1.0 and _auto_token_cap < 100:
        _auto_token_cap = 100
        TOKEN_CAP_GAUGE.set(100)
        _remediate("Token cap restored → 100", f"avg latency {w_mean:.2f}s — system healthy")

def _auto_throttle(recent: list, err_rate: float):
    global _throttle_delay
    if len(recent) < 10:
        return
    if err_rate >= 0.4 and _throttle_delay < 1.0:
        _throttle_delay = 1.0
        THROTTLE_GAUGE.set(1.0)
        _remediate("Throttle +1.0s/req", f"error rate {err_rate * 100:.0f}% ≥ 40% — slowing intake")
    elif err_rate >= 0.2 and _throttle_delay < 0.5:
        _throttle_delay = 0.5
        THROTTLE_GAUGE.set(0.5)
        _remediate("Throttle +0.5s/req", f"error rate {err_rate * 100:.0f}% ≥ 20% — slowing intake")
    elif err_rate < 0.05 and _throttle_delay > 0:
        _throttle_delay = 0.0
        THROTTLE_GAUGE.set(0.0)
        _remediate("Throttle removed", f"error rate recovered to {err_rate * 100:.0f}%")

def _auto_restart_vllm():
    global _vllm_process
    _remediate("vLLM restart initiated", "circuit breaker opened — attempting auto-recovery")
    try:
        if _vllm_process and _vllm_process.poll() is None:
            _vllm_process.terminate()
            try:
                _vllm_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _vllm_process.kill()

        _vllm_process = subprocess.Popen(
            VLLM_CMD,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _remediate(f"vLLM process started (PID {_vllm_process.pid})",
                   "waiting for model to load — probing health endpoint")
        threading.Thread(target=_probe_until_healthy, daemon=True).start()

    except FileNotFoundError:
        _alert("CRITICAL", "restart_failed",
               "Auto-restart FAILED — vLLM command not found. Start vLLM manually.")
    except Exception as exc:
        _alert("CRITICAL", "restart_failed", f"Auto-restart FAILED: {exc}")


def _probe_until_healthy():
    for attempt in range(24):
        datetime.time.sleep(5)
        try:
            r = requests.get(VLLM_HEALTH, timeout=3)
            if r.status_code == 200:
                cb_record_success()  # use abstraction — resets failures + Prometheus gauges
                _remediate(
                    "Circuit CLOSED — vLLM recovered",
                    f"health probe succeeded after {(attempt + 1) * 5}s — resuming normal routing"
                )
                return
        except Exception:
            pass
    _alert("CRITICAL", "recovery_timeout",
           "vLLM did not recover after 2 minutes — manual intervention required")


