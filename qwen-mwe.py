import asyncio
from pathlib import Path
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
import os


async def main():
    utils.setup_environment()
    API = InferenceAPI(cache_dir=Path(".cache"), openai_base_url="https://openrouter.ai/api/v1", openai_api_key=os.getenv("OPENROUTER_API_KEY"))
    prompt = Prompt(messages=[ChatMessage(content="Why do humans fight?", role=MessageRole.user)])
    response = await API(
        model_id="Qwen/Qwen3-8b",
        prompt=prompt,
        max_tokens=100,
        print_prompt_and_response=True,
        temperature=0,
        force_provider="openai",
    )
    print("Response:")
    print(response[0].completion)


if __name__ == "__main__":
    asyncio.run(main())
