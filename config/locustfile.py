import random

from locust import HttpUser, between, task

PROMPTS = [
    "AI is transforming the world by",
    "The future of machine learning is",
    "Once upon a time in a distant galaxy",
    "The key to success in software engineering",
    "Natural language processing helps computers",
    "Deep learning models are powerful because",
    "Climate change is a serious problem that",
    "The history of artificial intelligence began",
    "In 2050 the most important technology will be",
    "Python is the best programming language for",
]

class LLMUser(HttpUser):
    """Standard user — moderate pacing."""
    wait_time = between(1, 3)

    @task(5)
    def predict_short(self):
        prompt = random.choice(PROMPTS)
        self.client.get(
            f"/predict?text={prompt}&max_tokens=30",
            name="/predict [short]",
        )

    @task(2)
    def predict_long(self):
        prompt = random.choice(PROMPTS)
        self.client.get(
            f"/predict?text={prompt}&max_tokens=100",
            name="/predict [long]",
        )

    @task(1)
    def check_health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def fetch_stats(self):
        self.client.get("/api/stats", name="/api/stats")


class HeavyUser(HttpUser):
    """Aggressive user — drives up latency and error metrics."""
    wait_time = between(0.1, 0.5)

    @task
    def hammer_predict(self):
        prompt = random.choice(PROMPTS)
        self.client.get(
            f"/predict?text={prompt}&max_tokens=50",
            name="/predict [heavy]",
        )
