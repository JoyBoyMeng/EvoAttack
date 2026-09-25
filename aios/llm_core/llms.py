# This file just wraps around the LLM classes in aios/llm_kernel/llm_classes
# and provides an easy interface for the rest of the code to access
# All abstractions will be implemented here

from typing import Optional

from .llm_classes.model_registry import MODEL_REGISTRY
from .llm_classes.hf_native_llm import HfNativeLLM
from .llm_classes.gpt_llm import GPTLLM

# standard implementation of LLM methods
from .llm_classes.ollama_llm import OllamaLLM
from .llm_classes.vllm import vLLM

class LLMKernel:
    def __init__(self,
                 llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 256,
                 log_mode: str = "console",
                 use_backend: str = None,
                 enable_thinking: Optional[bool] = None,
        ):

        # For API-based LLM
        if llm_name in MODEL_REGISTRY.keys():
            model_class = MODEL_REGISTRY[llm_name]
            model_kwargs = dict(
                llm_name = llm_name,
                max_new_tokens = max_new_tokens,
                log_mode = log_mode
            )
            if issubclass(model_class, GPTLLM):
                model_kwargs["enable_thinking"] = enable_thinking
            self.model = model_class(**model_kwargs)
        # For locally-deployed LLM
        else:
            if use_backend == "ollama" or llm_name.startswith("ollama"):
                self.model = OllamaLLM(
                    llm_name=llm_name,
                    max_gpu_memory=max_gpu_memory,
                    eval_device=eval_device,
                    max_new_tokens=max_new_tokens,
                    log_mode=log_mode,
                    enable_thinking=enable_thinking,
                )

            elif use_backend == "vllm":
                self.model = vLLM(
                    llm_name=llm_name,
                    max_gpu_memory=max_gpu_memory,
                    eval_device=eval_device,
                    max_new_tokens=max_new_tokens,
                    log_mode=log_mode,
                    enable_thinking=enable_thinking,
                )
            else: # use huggingface LLM without backend
                self.model = HfNativeLLM(
                    llm_name=llm_name,
                    max_gpu_memory=max_gpu_memory,
                    eval_device=eval_device,
                    max_new_tokens=max_new_tokens,
                    log_mode=log_mode,
                    enable_thinking=enable_thinking,
                )

    def address_request(self,
                        agent_process,
                        temperature=0.0):
        self.model.address_request(agent_process,temperature)

    def address_request_list(self,
                        agent_process,
                        temperature=0.0):
        self.model.address_request_list(agent_process,temperature)
