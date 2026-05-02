import os
import time
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- API KEYS ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")

genai.configure(api_key=GEMINI_API_KEY)


class LLM:
    def __init__(
        self,
        temperature: float = 0.4,
        timeout: int = 10,
        max_output_tokens: int = 512
    ):
        self.temperature = temperature
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

        # Model tiers
        self.gemini_models = [
            "gemini-3-flash-preview",
            "gemini-2.5-flash"
        ]

        self.groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]

        self.groq_client = Groq(api_key=GROQ_API_KEY)

    # ---------------- GEMINI ---------------- #
    def _call_gemini(self, model_name, prompt, temperature=0.4):
        try:
            model = genai.GenerativeModel(model_name=model_name)

            start = time.time()

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature if temperature is not None else self.temperature,
                    "max_output_tokens": self.max_output_tokens
                }
            )

            latency = time.time() - start
            print(f"[Gemini:{model_name}] {latency:.2f}s")

            if latency > self.timeout:
                raise TimeoutError("Gemini timeout")

            if not response or not response.text:
                raise ValueError("Empty Gemini response")

            return response.text.strip()

        except Exception as e:
            print(f"[GEMINI FAIL - {model_name}] {str(e)}")
            raise

    # ---------------- GROQ ---------------- #
    def _call_groq(self, model_name, prompt, temperature=0.4):
        try:
            start = time.time()

            response = self.groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature if temperature is not None else self.temperature,
            )

            latency = time.time() - start
            print(f"[Groq:{model_name}] {latency:.2f}s")

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[GROQ FAIL - {model_name}] {str(e)}")
            raise

    # ---------------- CORE ROUTER ---------------- #
    def _generate_with_fallback(self, prompt, temperature=None):
        # Step 1 → Groq primary
        try:
            return self._call_groq(self.groq_models[0], prompt, temperature)
        except:
            print("[STEP] Groq primary failed → Gemini primary")

        # Step 2 → Gemini primary
        try:
            return self._call_gemini(self.gemini_models[0], prompt, temperature)
        except:
            print("[STEP] Gemini primary failed → Groq fallback")

        # Step 3 → Groq fallback
        try:
            return self._call_groq(self.groq_models[1], prompt, temperature)
        except:
            print("[STEP] Groq fallback failed → Gemini fallback")

        # Step 4 → Gemini fallback
        try:
            return self._call_gemini(self.gemini_models[1], prompt, temperature)
        except:
            print("[STEP] All providers exhausted")

        return "All models failed. Try again later."

    # ---------------- PUBLIC METHODS ---------------- #

    def generate(self, prompt: str) -> str:
        """
        Main generation (used by rag_core)
        """
        return self._generate_with_fallback(prompt, temperature=self.temperature)

    def rewrite(self, prompt: str) -> str:
        """
        Deterministic rewrite for query normalization
        """
        return self._generate_with_fallback(prompt, temperature=0.0)


# -------- Singleton (used everywhere) -------- #
_llm_instance = LLM()


def generate_response(prompt: str) -> str:
    return _llm_instance.generate(prompt)


def get_llm():
    """
    Optional accessor if you want to pass LLM into prompt_builder later
    """
    return _llm_instance


