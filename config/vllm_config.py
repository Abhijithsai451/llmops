import requests
from vllm.envs import VLLM_HOST_IP

VLLM_URL = "http://localhost:8001/v1/completions"
VLLM_HEALTH = "http://localhost:8001/health"
API_KEY = "some-random-text-only-for-local-testing"
session = requests.Session()

VLLM_CMD = [
    "python","-m","vllm.entrypoints.openai.api_server",
    "--model","./gpt2-finetuned",
    "--port","8001",
    "--api-key", API_KEY,
]


