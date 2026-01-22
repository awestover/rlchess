import asyncio
from pathlib import Path
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
import os

# URL = "http://localhost:8000/v1"
# MODEL = "Qwen/Qwen3-8B"
URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-oss-20b"

PROMPT = """
FEN: 1nbk4/1r3pp1/1p2p3/1p3r2/3N3P/b2PP1RP/P4PB1/RN1K4 w - - 7 28

Here's a chess board. Output a good move WITHOUT THINKING ABOUT IT FIRST.
"""

async def main():
    utils.setup_environment()
    API = InferenceAPI(cache_dir=Path(".cache"), openai_base_url=URL, openai_api_key=os.getenv("OPENROUTER_API_KEY"))
    prompt = Prompt(messages=[ChatMessage(content=PROMPT, role=MessageRole.user)])
    response = await API(
        model_id=MODEL,
        prompt=prompt,
        temperature=1,
        force_provider="openai",
        extra_body={
            "reasoning": {"effort": "medium"}
        },
        seed=42
    )
    print("Response:")
    print(response[0].completion)

    print("\n\n\n\n REASONING \n\n\nn")
    print(response[0].generated_content[0].content['message'].get('reasoning', []))

if __name__ == "__main__":
    asyncio.run(main())
