import statistics
from datetime import time, datetime

from fastapi import FastAPI, Query, requests
from fastapi.responses import Response, HTMLResponse, JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from config.alerting import _analyze, _alert
from config.local_model_inference import _local_infer
from config.remediation import  remediation_log
from config.circuit_breaker import cb_is_open, cb_record_success, cb_record_failure
from config.configs_params import (request_history, ACTIVE_REQUESTS, aiops_alerts, _next_id, _force_slow, _slow_seconds,
    _force_fail, LATENCY, ERROR_COUNT, REQUEST_COUNT, _inc_totals, TOKEN_COUNT, THROUGHPUT, _lock, CIRCUIT_OPEN,
    BACKEND_HEALTH, TOKEN_CAP_GAUGE, THROTTLE_GAUGE, _alert_lock)
from config.vllm_config import session, VLLM_URL, API_KEY

app = FastAPI(title="LLM AI Ops Platform", version="0.1.0")

@app.get("/")
def home():
    return {
        "service": "LLM AIOps Platform",
        "status": "running",
        "model": "gpt2-finetuned",
        "endpoints": {
            "predict": "/predict?text=<prompt>&max_tokens=50",
            "health": "/health",
            "metrics": "/metrics",
            "dashboard": "/dashboard",
            "stats": "/api/stats",
            "history": "/api/history",
        },
    }

@app.get("/health")
def health():
    recent = list(request_history)[:20]
    errors = [r for r in recent if r["status"] == "error"]
    success = [r for r in recent if r["status"] == "success"]
    lats = [r["latency"] for r in success]

    return {
        "status": "degraded" if cb_is_open() else "healthy",
        "circuit_breaker": "open" if cb_is_open() else "closed",
        "vllm_backend": "down" if cb_is_open() else "up",
        "active_requests": int(ACTIVE_REQUESTS._value.get()),
        "last_20_requests": {
            "total": len(recent),
            "errors": len(errors),
            "error_rate": f"{len(errors) / max(len(recent), 1) * 100:.1f}%",
            "avg_latency": f"{statistics.mean(lats):.2f}s" if lats else "n/a",
        },
        "open_alerts": len([a for a in aiops_alerts if a["severity"] in ("WARNING", "CRITICAL")]),
    }
@app.get("/predict")
def predict(
        text: str = Query(..., description = "input prompt"),
        max_tokens: int = Query(50, description = "max tokens to generate", ge=1, le=500),
    ):
    req_id = _next_id()
    backend_used = "vllm"
    generated = ""
    tokens = 0
    error_type = None
    latency = 0.0

    ACTIVE_REQUESTS.inc()
    start = time.time()

    if _throttle_delay > 0:
        time.sleep(_throttle_delay)
    if max_tokens > _auto_token_cap:
        max_tokens = _auto_token_cap
    if _force_slow:
        time.sleep(_slow_seconds)

    if _force_fail:
        latency = round(time.time() - start, 3)
        ACTIVE_REQUESTS.dec()
        LATENCY.observe(latency)
        ERROR_COUNT.labels(type="injected").inc()
        REQUEST_COUNT.labels(status="error").inc()
        _inc_totals(success=False)
        _analyze(latency)
        request_history.appendleft({
            "id": req_id, "timestamp": datetime.now().strftime("%H:%M:%S"),
            "prompt": text[:100], "tokens": 0, "latency": latency,
            "backend": "vllm", "status": "error", "error": "injected_failure",
        })
        return JSONResponse(status_code=503, content={"error": "injected_failure", "req_id": req_id})

    try:
        if cb_is_open():
            backend_used = "local_fallback"
            _alert("INFO", "using_fallback",
                   "Circuit open — using local transformers inference as fallback")
            generated, tokens = _local_infer(text, max_tokens)
            REQUEST_COUNT.labels(status="success").inc()

        else:
            resp = session.post(
                VLLM_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": "./gpt2-finetuned", "prompt": text, "max_tokens": max_tokens},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            generated = data["choices"][0]["text"]
            tokens = data.get("usage", {}).get("completion_tokens", len(generated.split()))
            cb_record_success()
            REQUEST_COUNT.labels(status="success").inc()

        TOKEN_COUNT.inc(tokens)

    except requests.exceptions.Timeout:
        error_type = "timeout"
        ERROR_COUNT.labels(type="timeout").inc()
        REQUEST_COUNT.labels(status="error").inc()
        cb_record_failure("timeout")

    except requests.exceptions.ConnectionError:
        error_type = "connection_error"
        ERROR_COUNT.labels(type="connection").inc()
        REQUEST_COUNT.labels(status="error").inc()
        cb_record_failure("connection_error")

    except Exception as exc:
        error_type = type(exc).__name__
        ERROR_COUNT.labels(type="unknown").inc()
        REQUEST_COUNT.labels(status="error").inc()
        cb_record_failure(str(exc)[:80])

    finally:
        latency = round(time.time() - start, 3)
        LATENCY.observe(latency)
        ACTIVE_REQUESTS.dec()
        if tokens > 0 and latency > 0:
            THROUGHPUT.set(tokens / latency)

        _inc_totals(success=error_type is None)
        _analyze(latency)

        request_history.appendleft({
            "id": req_id,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "prompt": text[:100],
            "tokens": tokens,
            "latency": latency,
            "backend": backend_used,
            "status": "error" if error_type else "success",
            "error": error_type,
        })

    if error_type and backend_used == "vllm":
        return JSONResponse(
            status_code=503,
            content={"error": error_type, "message": "vLLM backend unavailable.", "req_id": req_id},
        )

    return {
        "req_id": req_id,
        "prompt": text,
        "generated_text": generated,
        "tokens": tokens,
        "latency_seconds": latency,
        "backend": backend_used,
    }

@app.get("/api/history")
def api_history(limit: int = Query(50, ge=1, le=200)):
    return {"requests": list(request_history)[:limit]}


# ═══════════════════════════════════════════════════════════════════════════════
# Demo / Chaos Control
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/demo/fail/on")
def demo_fail_on():
    global _force_fail
    _force_fail = True
    _alert("CRITICAL", "demo_fail", "DEMO MODE: force-fail enabled — all /predict → 503")
    return {"chaos": "force_fail=ON"}

@app.post("/demo/fail/off")
def demo_fail_off():
    global _force_fail
    _force_fail = False
    _alert("INFO", "demo_fail_off", "DEMO MODE: force-fail disabled")
    return {"chaos": "force_fail=OFF"}

@app.post("/demo/slow/on")
def demo_slow_on(seconds: float = Query(3.0, ge=0.5, le=30.0)):
    global _force_slow, _slow_seconds
    _force_slow   = True
    _slow_seconds = seconds
    _alert("WARNING", "demo_slow", f"DEMO MODE: latency injection +{seconds}s per request")
    return {"chaos": "force_slow=ON", "added_latency_seconds": seconds}

@app.post("/demo/slow/off")
def demo_slow_off():
    global _force_slow
    _force_slow = False
    _alert("INFO", "demo_slow_off", "DEMO MODE: latency injection disabled")
    return {"chaos": "force_slow=OFF"}

@app.post("/demo/reset")
def demo_reset():
    global _force_fail, _force_slow, _cb_open, _cb_failures
    global _auto_token_cap, _throttle_delay
    global _total_requests, _total_success, _total_errors
    with _lock:
        _force_fail     = False
        _force_slow     = False
        _cb_open        = False
        _cb_failures    = 0
        _auto_token_cap = 100
        _throttle_delay = 0.0
        _total_requests = 0
        _total_success  = 0
        _total_errors   = 0
    CIRCUIT_OPEN.set(0)
    BACKEND_HEALTH.set(1)
    TOKEN_CAP_GAUGE.set(100)
    THROTTLE_GAUGE.set(0.0)
    with _alert_lock:
        aiops_alerts.clear()
    request_history.clear()
    remediation_log.clear()
    _alert("INFO", "demo_reset", "DEMO RESET: all chaos + remediation state cleared")
    return {"status": "reset"}

@app.get("/demo/status")
def demo_status():
    return {
        "force_fail":   _force_fail,
        "force_slow":   _force_slow,
        "slow_seconds": _slow_seconds,
        "circuit_open": _cb_open,
        "cb_failures":  _cb_failures,
    }


@app.get("/api/remediation")
def api_remediation():
    return {
        "current": {
            "token_cap":      _auto_token_cap,
            "throttle_delay": _throttle_delay,
            "circuit_open":   _cb_open,
        },
        "log": list(remediation_log),
    }


@app.get("/api/stats")
def api_stats():
    history     = list(request_history)
    lats        = [r["latency"] for r in history if r["status"] == "success"]
    chart_items = list(reversed(history[:30]))
    return {
        "metrics": {
            "total_requests":  _total_requests,
            "success_count":   _total_success,
            "error_count":     _total_errors,
            "success_rate":    round(_total_success / max(_total_requests, 1) * 100, 1),
            "avg_latency":     round(statistics.mean(lats), 3) if lats else 0,
            "p95_latency":     round(sorted(lats)[int(len(lats) * 0.95)], 3) if len(lats) > 1 else 0,
            "total_tokens":    sum(r["tokens"] for r in history),
            "active_requests": int(ACTIVE_REQUESTS._value.get()),
        },
        "system": {
            "circuit_breaker": "open" if cb_is_open() else "closed",
            "backend_healthy": not cb_is_open(),
            "backend":         "local_fallback" if cb_is_open() else "vLLM",
        },
        "demo_status": {           # merged here — eliminates separate /demo/status poll
            "force_fail":   _force_fail,
            "force_slow":   _force_slow,
            "slow_seconds": _slow_seconds,
            "cb_failures":  _cb_failures,
        },
        "charts": {
            "labels":   [r["timestamp"] for r in chart_items],
            "latencies": [r["latency"]  for r in chart_items],
            "statuses":  [r["status"]   for r in chart_items],
            "tokens":    [r["tokens"]   for r in chart_items],
        },
        "alerts":          list(aiops_alerts)[:15],
        "recent_requests": history[:20],
        "remediation": {
            "token_cap":      _auto_token_cap,
            "throttle_delay": _throttle_delay,
            "log":            list(remediation_log)[:8],
        },
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    with open("templates/dashboard.html", "r") as f:
        return f.read()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)




