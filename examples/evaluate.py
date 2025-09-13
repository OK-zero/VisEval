# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from pathlib import Path
from os import getenv

import dotenv
from agent import Chat2vis, CoML4VIS, Lida

from viseval import Dataset, Evaluator
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


model_name_to_id = {
    "qwen3-4b": "qwen/qwen3-4b:free",
    "qwen3-8b": "qwen/qwen3-8b:free",
    "qwen3-14b": "qwen/qwen3-14b:free",
    "qwen2.5": "qwen2.5-coder-7b-instruct",
}


def configure_llm(model: str):
    if model in model_name_to_id.keys():
        return ChatOpenAI(
            model_name=model_name_to_id[model],
            base_url=getenv("BASE_URL"),
            max_retries=999,
            temperature=0.0,
            request_timeout=20,
            max_tokens=4096,
        )
    else:
        raise ValueError(f"Unknown model {model}")


def config_agent(agent: str, model: str, config: dict):
    llm = configure_llm(model)
    if agent == "coml4vis":
        return CoML4VIS(llm, config)
    elif agent == "chat2vis":
        return Chat2vis(llm)
    elif agent == "lida":
        return Lida(llm)
    else:
        raise ValueError(f"Unknown agent {agent}")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument(
        "--type",
        type=str,
        choices=["all", "single", "multiple", "sample"],
        default="all",
    )
    parser.add_argument("--irrelevant_tables", type=bool, default=False)

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5",
        choices=["qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen2.5"],
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="coml4vis",
        choices=["coml4vis", "lida", "chat2vis"],
    )
    parser.add_argument("--num_examples", type=int, default=1, choices=range(0, 4))
    parser.add_argument(
        "--library", type=str, default="matplotlib", choices=["matplotlib", "seaborn"]
    )
    parser.add_argument("--logs", type=Path, default="./logs")
    parser.add_argument(
        "--webdriver", type=Path, default="/opt/homebrew/bin/chromedriver"
    )

    args = parser.parse_args()

    # config dataset
    dataset = Dataset(args.benchmark, args.type, args.irrelevant_tables)

    # config agent
    agent = config_agent(
        args.agent,
        args.model,
        {"num_examples": args.num_examples, "library": args.library},
    )

    vision_model = ChatOpenAI(
        model_name="google/gemma-3-4b-it:free",
        base_url=getenv("OPENROUTER_BASE_URL"),
        max_retries=999,
        temperature=0.0,
        request_timeout=20,
        max_tokens=4096,
    )
    # config evaluator
    evaluator = Evaluator(webdriver_path=args.webdriver, vision_model=vision_model)

    # evaluate agent
    config = {
        "library": args.library,
        "logs": Path(
            f"{args.logs}_{args.agent}_{args.model}_{args.type}_{args.library}"
        ),
    }
    result = evaluator.evaluate(agent, dataset, config)

    # write result to logs
    result.save(config["logs"])


if __name__ == "__main__":
    _main()
