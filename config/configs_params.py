# ─── Prometheus Metrics ──────────────────────────────────────────────────────
import datetime

from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT     = Counter("llm_requests_total",         "Total LLM requests",              ["status"])
ERROR_COUNT       = Counter("llm_errors_total",           "Total LLM errors",                ["type"])
TOKEN_COUNT       = Counter("llm_tokens_total",           "Total tokens generated")
LATENCY           = Histogram("llm_latency_seconds",      "Request latency",
                              buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
ACTIVE_REQUESTS   = Gauge("llm_active_requests",          "In-flight requests")
THROUGHPUT        = Gauge("llm_tokens_per_second",        "Token throughput (last req)")
BACKEND_HEALTH    = Gauge("llm_backend_healthy",          "1 = vLLM up, 0 = down")
CIRCUIT_OPEN      = Gauge("llm_circuit_breaker_open",     "1 = circuit open")
TOKEN_CAP_GAUGE   = Gauge("llm_auto_token_cap",           "Current auto token cap")
THROTTLE_GAUGE    = Gauge("llm_throttle_delay_seconds",   "Current throttle delay injected")
REMEDIATION_COUNT = Counter("llm_remediations_total",     "Auto-remediation actions taken",  ["action"])

# ─── In-memory state ─────────────────────────────────────────────────────────
import threading
from collections import deque

request_history: deque = deque(maxlen=200)
_total_requests  = 0
_total_success   = 0
_total_errors    = 0
latency_window:  deque = deque(maxlen=20)
aiops_alerts:    deque = deque(maxlen=100)
_last_alert_key: dict  = {}

# ─── Demo / Chaos flags ───────────────────────────────────────────────────────
_force_fail   = False
_force_slow   = False
_slow_seconds = 3.0


# ─── Locks ───────────────────────────────────────────────────────────────────
_lock       = threading.Lock()   # circuit breaker, request counter, totals
_alert_lock = threading.Lock()   # alert dedup dict + deque (separate to avoid reentrant deadlock)




