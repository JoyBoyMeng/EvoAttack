from __future__ import annotations

import argparse
import json
from typing import Any

from dotenv import load_dotenv

from attack_agent.local_model_config import make_openai_client


def serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether OpenAI chat completions return top_logprobs.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--top_logprobs", type=int, default=1)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass enable_thinking in extra_body. Qwen3 non-streaming APIs usually require --no-enable_thinking.",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=".env")
    client = make_openai_client(api_key=args.api_key, base_url=args.base_url)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a constrained classifier. Output exactly one character and nothing else. "
                "Valid outputs are: 1, 2, or 3."
            ),
        },
        {
            "role": "user",
            "content": (
                "Candidate strategies:\n"
                "1. Capability-Task Alignment\n"
                "2. Verification and Reliability Framing\n"
                "3. Context Completion\n\n"
                "Choose the best candidate strategy. Answer with exactly one character: 1 or 2 or 3"
            ),
        },
    ]

    for attempt in range(1, args.attempts + 1):
        request_kwargs = {}
        if args.enable_thinking is not None:
            request_kwargs["extra_body"] = {"enable_thinking": args.enable_thinking}

        completion = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=args.top_logprobs,
            **request_kwargs,
        )
        choice = completion.choices[0]
        content = choice.message.content
        logprob_content = choice.logprobs.content if choice.logprobs else []
        first = logprob_content[0] if logprob_content else None
        top = first.top_logprobs if first else []
        rows = [
            {
                "token": str(item.token),
                "stripped": str(item.token).strip(),
                "logprob": float(item.logprob),
            }
            for item in top
        ]
        print("=" * 80)
        print(f"attempt: {attempt}/{args.attempts}")
        print(f"model: {args.model}")
        print(f"message_content: {content!r}")
        print(f"logprobs_present: {choice.logprobs is not None}")
        print(f"logprob_content_len: {len(logprob_content)}")
        print(f"top_logprobs_len: {len(rows)}")
        print("top_logprobs:")
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        print("raw_choice_logprobs:")
        print(json.dumps(serialize(choice.logprobs), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
