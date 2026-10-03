import httpx


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float = 45.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "stream": False,
                        "think": False,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 256,
                        },
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError("Local language model is unavailable.") from exc

        try:
            answer = response.json()["message"]["content"].strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Local language model returned an invalid response.") from exc
        if not answer:
            raise RuntimeError("Local language model returned an empty response.")
        return answer
