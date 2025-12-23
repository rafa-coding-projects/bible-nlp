from cls_models import QueryReformulator
from typing import List


class MultiQueryReformulator(QueryReformulator):
    """Generate multiple positive query variations."""

    def reformulate(self, query: str) -> List[str]:
        return [
            query,
            f"spiritual wisdom about {query}",
            f"guidance and comfort for {query}",
            f"teaching on {query}",
            f"hope and encouragement for {query}",
        ]


class LLMReformulator(QueryReformulator):
    """Use LLM to intelligently reformulate queries."""

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"
    ):
        self.base_url = base_url
        self.model = model

    def reformulate(self, query: str) -> List[str]:
        import requests

        prompt = f"""Transform this into a positive, solution-focused search query for finding spiritual guidance.

Original: "{query}"

Rules:
- Focus on solutions, wisdom, and positive outcomes
- Keep concise (under 15 words)
- Use words like: wisdom, guidance, peace, comfort, strength, hope, faith, love

Reformulated query:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "max_tokens": 50},
                },
                timeout=10,
            )

            if response.status_code == 200:
                reformulated = response.json()["response"].strip().strip('"')
                return [query, reformulated]  # Original + reformulated
        except:
            pass

        # Fallback
        return [query, f"spiritual guidance for {query}"]
