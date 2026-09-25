from .base_llm import BaseLLM
import time

# could be dynamically imported similar to other models
from openai import OpenAI

import openai

from pyopenagi.utils.chat_template import Response
import json
import os
from typing import Optional

from aios.llm_core.thinking import openai_thinking_extra_body


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def token_budgets(initial: int, maximum: int):
    budgets = []
    value = max(1, int(initial))
    maximum = max(value, int(maximum))
    while value < maximum:
        budgets.append(value)
        value *= 2
    budgets.append(maximum)
    return budgets


class GPTLLM(BaseLLM):

    def __init__(self, llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 1024,
                 log_mode: str = "console",
                 enable_thinking: Optional[bool] = None):
        if enable_thinking is None:
            enable_thinking = env_bool("TARGET_ENABLE_THINKING", False)
        super().__init__(llm_name,
                         max_gpu_memory,
                         eval_device,
                         max_new_tokens,
                         log_mode,
                         enable_thinking)

    def load_llm_and_tokenizer(self) -> None:
        # self.model = OpenAI()
        # self.tokenizer = None
        self.model = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.tokenizer = None

    def _extra_body(self):
        return openai_thinking_extra_body(
            model_name=self.model_name,
            enable_thinking=bool(self.enable_thinking),
        )

    def _max_tokens_limit(self) -> int:
        return int(os.getenv("TARGET_MAX_TOKENS_LIMIT", "16384"))

    def parse_tool_calls(self, tool_calls):
        if tool_calls:
            parsed_tool_calls = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                parsed_tool_calls.append(
                    {
                        "name": function_name,
                        "parameters": function_args
                    }
                )
            return parsed_tool_calls
        return None

    def process(self,
            agent_process,
            temperature=0.0
        ):
        """ wrapper around OpenAI or OpenAI-compatible chat completion API """
        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())
        messages = agent_process.query.messages
        # print(messages)
        self.logger.log(
            f"{agent_process.agent_name} is switched to executing.\n",
            level = "executing"
        )
        time.sleep(2)
        try:
            extra_body = self._extra_body()
            max_tokens_limit = self._max_tokens_limit() if self.enable_thinking else self.max_new_tokens
            last_response = None
            for max_tokens in token_budgets(self.max_new_tokens, max_tokens_limit):
                request = {
                    "model": self.model_name,
                    "messages": messages,
                    "tools": agent_process.query.tools,
                    "tool_choice": "required" if agent_process.query.tools else None,
                    "max_tokens": max_tokens,
                    "seed": 0,
                    "temperature": temperature,
                }
                if extra_body is not None:
                    request["extra_body"] = extra_body
                response = self.model.chat.completions.create(**request)
                last_response = response
                message = response.choices[0].message
                response_message = message.content
                tool_calls = self.parse_tool_calls(message.tool_calls)

                if tool_calls or (response_message and str(response_message).strip()):
                    agent_process.set_response(
                        Response(
                            response_message=response_message,
                            tool_calls=tool_calls
                        )
                    )
                    break

                print(
                    f"[GPTLLM] empty content/tool_calls at max_tokens={max_tokens}; increasing output budget",
                    flush=True,
                )
            else:
                reasoning = getattr(last_response.choices[0].message, "reasoning", None) if last_response else None
                agent_process.set_response(
                    Response(
                        response_message=(
                            "OpenAI-compatible response content remained empty after "
                            f"increasing max_tokens to {max_tokens_limit}. "
                            f"reasoning={reasoning!r}"
                        ),
                        tool_calls=None,
                    )
                )
        except openai.APIConnectionError as e:
            agent_process.set_response(
                Response(
                    response_message = f"Server connection error: {e.__cause__}"
                )
            )
        except openai.RateLimitError as e:
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI RATE LIMIT error {e.status_code}: (e.response)"
                )
            )
        except openai.APIStatusError as e:
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI STATUS error {e.status_code}: (e.response)"
                )
            )
        except openai.BadRequestError as e:
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI BAD REQUEST error {e.status_code}: (e.response)"
                )
            )
        except Exception as e:
            agent_process.set_response(
                Response(
                    response_message = f"An unexpected error occurred: {e}"
                )
            )

        agent_process.set_status("done")
        agent_process.set_end_time(time.time())
