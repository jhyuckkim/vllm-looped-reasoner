## Installation

First, have vLLM installed in the virtual environment that you are using.
E.g.,
```bash
cd tiny-reasoner
source .venv/bin/activate
uv pip install vllm
```

Then clone this repository and install it in editable mode in the same environment.

```bash
cd ..
git clone https://github.com/jhyuckkim/vllm-looped-reasoner.git
cd vllm-looped-reasoner
uv pip install -e .
```

Now you can use this environment to load Reasoner/LoopedReasoner HF checkpoints with vLLM.
E.g.,
```python
from vllm import LLM, SamplingParams

model_id = "LoopedReasoner-Qwen3-0.6B-Overfit"

llm = LLM(model=model_id, dtype="bfloat16", trust_remote_code=True)

sampling_params = SamplingParams(temperature=0.0, max_tokens=30)

prompts = [
    "Hello,",
    "I use",
    "With loops",
    "The weather",
]
outputs = llm.generate(prompts, sampling_params)
```
