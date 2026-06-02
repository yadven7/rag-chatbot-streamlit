import os
import time
from typing import Generator as TypingGenerator

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None

try:
    import ollama
except ImportError:
    ollama = None

class Generator:
    """
    Interfaces with the Google Gemini API to generate responses
    grounded strictly in the retrieved text context passages.
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.has_api = False
        
        # Configure Gemini API if key is present and not the default placeholder
        if genai is not None and self.api_key and self.api_key != "your_google_gemini_api_key_here":
            try:
                genai.configure(api_key=self.api_key)
                self.has_api = True
                print("Gemini API configured successfully.")
            except Exception as e:
                print(f"Error configuring Gemini API: {e}")

    def construct_prompt(self, query: str, retrieved_chunks: list) -> str:
        """
        Structures the retrieval contexts alongside grounding instructions
        and the user's query into a prompt.
        """
        context_str = ""
        for idx, chunk in enumerate(retrieved_chunks):
            context_str += f"\n[Source {idx+1}]\n{chunk.get('text', '')}\n"

        prompt = (
            "System Instructions:\n"
            "You are a helpful, precise, and fact-focused AI Assistant. "
            "Your task is to answer the user's query based ONLY on the provided Context Sources. "
            "Strictly adhere to the following rules:\n"
            "1. Rely only on the clear facts mentioned in the Context. Do not make up facts, external details or assumptions.\n"
            "2. If the answer cannot be found in the provided Context, say: 'I apologize, but the provided documents do not contain the answer to your query.'\n"
            "3. Cite which Context Source(s) you used to construct your answer (e.g., [Source 1], [Source 2]).\n"
            "4. Keep your response factual, grounded, and concise.\n\n"
            f"Context Sources:\n{context_str}\n\n"
            f"User Query: {query}\n\n"
            "Grounded Answer:"
        )
        return prompt

    def _generate_ollama_stream(self, prompt: str) -> TypingGenerator[str, None, None]:
        """
        Streams response from the local Ollama LLM model llama3.2:3b.
        """
        if ollama is None:
            yield "❌ **Error: Ollama python library is not installed.**\n\nPlease install it using: `pip install ollama`"
            return

        try:
            response = ollama.generate(
                model="llama3.2:3b",
                prompt=prompt,
                stream=True
            )
            for chunk in response:
                yield chunk.get("response", "")
        except Exception as e:
            err_str = str(e)
            if "not found" in err_str.lower() or "404" in err_str:
                yield (
                    "❌ **Error: Ollama model 'llama3.2:3b' not found.**\n\n"
                    "Please download and pull the model by running the following command in Windows Command Prompt or PowerShell:\n"
                    "```cmd\n"
                    "ollama pull llama3.2:3b\n"
                    "```"
                )
            else:
                yield (
                    f"❌ **Error connecting to Ollama: {e}**\n\n"
                    "Please make sure Ollama is installed and running.\n"
                    "To start the Ollama server and run the model, use the following commands in Windows:\n"
                    "```cmd\n"
                    "ollama serve\n"
                    "```\n"
                    "And in another terminal window, ensure the model is pulled:\n"
                    "```cmd\n"
                    "ollama pull llama3.2:3b\n"
                    "```"
                )

    def generate_stream(self, query: str, retrieved_chunks: list, model_name: str = "Ollama llama3.2:3b") -> TypingGenerator[str, None, None]:
        """
        Streams generated answers from either the Gemini model or Ollama model.
        If Gemini fails with a 429 quota error, it automatically falls back to Ollama.
        """
        prompt = self.construct_prompt(query, retrieved_chunks)

        # Check if we should use Gemini
        use_gemini = "gemini" in model_name.lower()

        if use_gemini:
            if self.has_api and genai is not None:
                try:
                    # Map the display name to the correct model identifier
                    api_model_name = "gemini-2.0-flash"
                    if "2.5" in model_name:
                        api_model_name = "gemini-2.5-flash"
                    elif "2.0" in model_name:
                        api_model_name = "gemini-2.0-flash"

                    model = genai.GenerativeModel(api_model_name)
                    response = model.generate_content(prompt, stream=True)
                    
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                    return
                except Exception as e:
                    # Detect 429 / Quota error
                    is_quota_error = False
                    if google_exceptions and isinstance(e, google_exceptions.ResourceExhausted):
                        is_quota_error = True
                    elif "429" in str(e) or "quota" in str(e).lower() or "exhausted" in str(e).lower():
                        is_quota_error = True

                    if is_quota_error:
                        yield "⚠️ *[Gemini API 429 Quota Exceeded - Automatically switching to local Ollama llama3.2:3b]*\n\n"
                        yield from self._generate_ollama_stream(prompt)
                        return
                    else:
                        yield f"\n[Gemini Generation Error: {e}]\n"
                        return
            else:
                yield "⚠️ *[Gemini API Key missing or invalid - Automatically switching to local Ollama llama3.2:3b]*\n\n"
                yield from self._generate_ollama_stream(prompt)
                return
        else:
            # Direct Ollama path
            yield from self._generate_ollama_stream(prompt)


if __name__ == "__main__":
    generator = Generator()
    test_chunks = [{"text": "Amlgo Labs is a digital transformation agency focusing on Advanced Analytics, AI, and Big Data."}]
    print("Testing Stream Generation:")
    for token in generator.generate_stream("What is Amlgo Labs?", test_chunks):
        print(token, end="", flush=True)
    print()
