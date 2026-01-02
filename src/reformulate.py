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

    def reformulate(self, query: str) -> str:
        import requests

        prompt = f"""Transform this problem-focused question into a positive, solution-focused search query for finding helpful Bible verses.

Original question: "{query}"

Rules:
- Focus on solutions, wisdom, and positive outcomes rather than problems
- Keep it concise (under 15 words)
- Use words like: wisdom, guidance, peace, comfort, strength, hope, faith, love
- Remove negative framing

Examples:
- "dealing with difficult people at work" → "wisdom for peaceful relationships at work"
- "I'm afraid of failure" → "courage and faith in challenging times"
- "feeling lonely and sad" → "God's comfort and companionship"

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
                reformulated = response.json()['response'].strip().strip('"').strip()
                return reformulated
        except Exception as e:
            print(f"LLM reformulation error: {str(e)}")
            
        # Fallback
        return f"biblical wisdom for {query}"
