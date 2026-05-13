import datetime
import statistics

from config.remediation import _auto_throttle, _auto_adjust_token_cap

_ALERT_COOLDOWN = 20
from configs_params import _alert_lock, _last_alert_key, aiops_alerts, latency_window, request_history


def _alert(severity: str, key: str, message: str):
    with _alert_lock:
        now = datetime.now()
        last = _last_alert_key.get(key)
        if last and (now - last).seconds <_ALERT_COOLDOWN:
            return
        _last_alert_key[key] = now
        aiops_alerts.appendleft({
            "severity": severity,
            "message": message,
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "key": key
        })

def _analyze(latency: float):
    latency_window.append(latency)
    window = list(latency_window)
    w_len = len(window)

    w_mean = statistics.mean(window) if w_len>=5 else None

    if w_len>=8 and w_mean is not None:
        stdev = statistics.pstdev(window)
        if stdev>0:
            z= (latency - w_mean)/ stdev
            if z> 2.5:
                _alert("WARNING","latency_anamoly",
                    f"Latency spike: {latency:.2f}s is {z:.1f}σ above mean ({w_mean:.2f}s)")
    r# 2. Rolling error rate — single list copy shared with auto-throttle
    recent   = list(request_history)[:10]
    err_rate = sum(1 for r in recent if r["status"] == "error") / len(recent) if len(recent) >= 5 else 0.0
    if len(recent) >= 5:
        if err_rate >= 0.5:
            _alert("CRITICAL", "high_error_rate",
                   f"Error rate {err_rate*100:.0f}% in last {len(recent)} requests")
        elif err_rate >= 0.2:
            _alert("WARNING", "elevated_error_rate",
                   f"Elevated error rate {err_rate*100:.0f}% in last {len(recent)} requests")

    # 3. Sustained high latency
    if w_mean is not None and w_mean > 4.0:
        _alert("WARNING", "sustained_latency",
               f"Sustained high latency: avg {w_mean:.2f}s over last {w_len} requests")

    # pass pre-computed values to avoid redundant work
    _auto_adjust_token_cap(w_mean)
    _auto_throttle(recent, err_rate)
