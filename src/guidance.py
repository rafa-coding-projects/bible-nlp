from typing import List, Optional
from cls_models import SearchResult, GuidanceGenerator, GuidanceOutput
import requests


def check_llm_server(base_url='http://localhost:11434'):
    """Check if Ollama or compatible LLM server is running"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✓ LLM server is running at {base_url}")
            print(f"Available models: {[m['name'] for m in models]}")
            return True
    except:
        print(f"✗ No LLM server found at {base_url}")
        print("\nTo install Ollama on Jetson:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        print("  ollama pull llama3.2:3b  # or llama3.2:1b for faster inference")
        return False


class TextGuidanceGenerator(GuidanceGenerator):
    """Generate text guidance using LLM."""

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"
    ):
        self.base_url = base_url
        self.model = model
        check_llm_server()

    def generate(self, query: str, results: List[SearchResult]) -> GuidanceOutput:
        import requests

        # Format sources
        context = "\n\n".join(
            [f"From {r.source} ({r.reference}):\n{r.text}" for r in results]
        )

        prompt = f"""You are a wise spiritual counselor. Someone has asked:

"{query}"

Here are relevant teachings from sacred sources:

{context}

Provide a thoughtful, encouraging response that:
1. Do not validate negative feelings
2. Explains how these teachings relate to their concern
3. Offers specific, actionable guidance
4. Ends with hope and encouragement

Keep your response concise (3-4 paragraphs) and warm in tone."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "top_p": 0.9},
                },
                timeout=60,
            )

            if response.status_code == 200:
                guidance_text = response.json()["response"]
            else:
                guidance_text = "Error generating guidance."
        except Exception as e:
            guidance_text = f"Error: {str(e)}"

        return GuidanceOutput(query=query, results=results, guidance_text=guidance_text)


class MultimodalGuidanceGenerator(GuidanceGenerator):
    """Generate text guidance + illustrative image."""

    def __init__(
        self,
        text_generator: TextGuidanceGenerator,
        image_model: str = "stable-diffusion-xl",
        image_api_url: str = "http://localhost:7860",
    ):
        self.text_generator = text_generator
        self.image_model = image_model
        self.image_api_url = image_api_url

    def _generate_image_prompt(self, query: str, guidance_text: str) -> str:
        """Generate an image prompt from the guidance."""
        # TODO: Use LLM to create artistic prompt from guidance
        return (
            f"peaceful spiritual scene representing {query}, serene, inspirational art"
        )

    def _generate_image(self, prompt: str) -> Optional[str]:
        """Generate image and return URL or base64."""
        import requests

        try:
            # Example: Stable Diffusion API
            response = requests.post(
                f"{self.image_api_url}/sdapi/v1/txt2img",
                json={
                    "prompt": prompt,
                    "negative_prompt": "dark, violent, disturbing",
                    "steps": 20,
                    "width": 512,
                    "height": 512,
                },
                timeout=60,
            )

            if response.status_code == 200:
                # Return base64 or save and return URL
                image_data = response.json()["images"][0]
                return f"data:image/png;base64,{image_data}"
        except:
            pass

        return None

    def generate(self, query: str, results: List[SearchResult]) -> GuidanceOutput:
        # Generate text guidance first
        output = self.text_generator.generate(query, results)

        # Generate image prompt and image
        image_prompt = self._generate_image_prompt(query, output.guidance_text)
        image_url = self._generate_image(image_prompt)

        output.image_url = image_url
        output.image_prompt = image_prompt

        return output
