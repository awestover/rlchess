# Important notes
- Before running any experiments give me time and cost estimates.
- When doing LLM inference, please make sure to use ample amounts of concurrency!

# Inference

Please use the together.ai or fireworks.ai API for doing inference on 

Please use this library for doing inference on LLMs:
https://github.com/safety-research/safety-tooling

The code will be something like this

```python
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from pathlib import Path
utils.setup_environment()
API = InferenceAPI(cache_dir=Path(".cache"), openai_base_url="https://openrouter.ai/api/v1", openai_api_key=openrouter_api_key)
prompt = Prompt(messages=[ChatMessage(content="What is your name?", role=MessageRole.user)])
response = await API(
    model_id="deepseek/deepseek-v3-base:free",
    prompt=prompt,
    max_tokens=100,
    print_prompt_and_response=True,
    temperature=0,
    force_provider="openai",
)
```

Note that you'll need to install safetytooling via
```bash
pip install git+https://github.com/safety-research/safety-tooling.git@<main>#egg=safetytooling
```
