import argparse
import asyncio
import os
import re
from pathlib import Path

import nest_asyncio
import pandas as pd
import yaml
from datasets import Dataset
from openai import AsyncOpenAI
from tqdm import tqdm

nest_asyncio.apply()

ROOT_DIR = Path(__file__).resolve().parent.parent
QUESTION_DIR_MAP = {
    "summeval": "summeval_questions",
    "topical_chat": "topical_chat_questions",
    "ambiguity": "ambiguity_questions",
}


class vLLMProcessor:
    def __init__(self, api_key, base_url, model, batch=False, batch_size=5):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.batch = batch
        self.batch_size = batch_size

    async def chat_completion(self, client, input_data, **kwargs):
        messages = [{"role": "user", "content": input_data}]
        response = await client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return response

    async def run_chat_completions(self, client, prompts: list, **kwargs):
        tasks = [self.chat_completion(client, prompt, **kwargs) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        return responses

    async def process(self, prompts, **kwargs):
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            responses = await self.run_chat_completions(client, prompts, **kwargs)
        return responses


class OpenaiProcessor:
    def __init__(self, api_key, model, batch=False, batch_size=5):
        self.api_key = api_key
        self.model = model
        self.batch = batch
        self.batch_size = batch_size

    async def chat_completion(self, client, input_data, **kwargs):
        messages = [{"role": "user", "content": input_data}]
        response = await client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return response

    async def run_chat_completions(self, client, prompts: list, **kwargs):
        tasks = [self.chat_completion(client, prompt, **kwargs) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        return responses

    async def process(self, prompts, **kwargs):
        async with AsyncOpenAI(api_key=self.api_key) as client:
            responses = await self.run_chat_completions(client, prompts, **kwargs)
        return responses


def _normalize_questions(raw_questions):
    if raw_questions is None:
        return []
    if isinstance(raw_questions, list):
        candidates = [str(q).strip() for q in raw_questions if str(q).strip()]
    else:
        text = str(raw_questions).strip()
        if not text:
            return []
        if "?" in text:
            candidates = [chunk.strip() for chunk in re.split(r"\?\s*", text) if chunk.strip()]
        else:
            candidates = [line.strip("-• \t") for line in text.splitlines() if line.strip()]

    normalized = []
    for candidate in candidates:
        question = re.sub(r"\s+", " ", candidate).strip()
        if not question:
            continue
        if not question.endswith("?"):
            question = question.rstrip(".!") + "?"
        normalized.append(question)
    return normalized


def make_question_list(questions):
    if isinstance(questions, str):
        question_items = _normalize_questions(questions)
    else:
        question_items = []
        for raw_question in questions:
            question_items.extend(_normalize_questions(raw_question))
    return "\n".join([f"Q{idx}: {question}" for idx, question in enumerate(question_items, 1)])


def extract_answers(response):
    matches = re.findall(r"Q\d+: (Yes|No)", response)
    return [1 if answer == "Yes" else 0 for answer in matches]


def _split_into_batches(data, num_batches):
    batch_size = len(data) // num_batches
    batches = [data[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]
    if len(data) % num_batches != 0:
        batches.append(data[num_batches * batch_size :])
    return batches


summeval_template = '''### Task Overview ###\nYour task is to read a provided news article and its summary, then answer 'yes' or 'no' to specific questions. These questions will relate to a particular aspect of the summary.\n\n### Aspect Definition ###\n<aspect> - <definition>\n\n### Instructions ###\n1. Read these instructions thoroughly.\n2. Carefully read both the Article and the Summary.\n3. Understand the given questions and the definition of the <aspect>.\n4. Respond to each question with 'yes' or 'no'. Base your answers on a clear rationale.\n5. Follow the specified format for your answers.\n\n### Answer Format ###\nQ1: [Your Answer] \nQ2: [Your Answer] \n...\n\n# Article #\n"<source>"\n\n# Summary #\n"<summary>"\n\n# Questions # \n<questions>\n\n# Response\nProvide your answers to the given questions, following the specified Answer Format.\n'''


topical_chat_template = '''### Task Overview ###\nYou will be given a conversation between two individuals. You will then be given one potential response for the next turn in the conversation. The response concerns an interesting fact, which will be provided as well.\nYour task is to read a provided conversation history, corresponding fact and response, then answer 'yes' or 'no' to specific questions. These questions will relate to a particular aspect of the response.\n\n### Aspect Definition ###\n<aspect> - <definition>\n\n### Instructions ###\n1. Read these instructions thoroughly.\n2. Carefully read the Conversation History, the Corresponding Fact and the Response.\n3. Understand the given questions and the definition of the <aspect>.\n4. Respond to each question with 'yes' or 'no'. Base your answers on a clear rationale.\n5. Follow the specified format for your answers.\n\n### Answer Format ###\nQ1: [Your Answer] \nQ2: [Your Answer] \n...\n\n# Conversation History #\n"<document>"\n\n# Corresponding Fact #\n"<fact>"\n\n# Response #\n"<response>"\n\n# Questions # \n<questions>\n\n# Your answer\nProvide your answers to the given questions, following the specified Answer Format.\n'''


def resolve_question_path(template_type, aspect, question_version):
    filename = f"{aspect}_{question_version}.yaml"
    question_dir = QUESTION_DIR_MAP.get(template_type, template_type)

    candidates = [
        ROOT_DIR / "prompt" / question_dir / filename,
        ROOT_DIR / "prompt" / template_type / filename,
        ROOT_DIR / question_dir / filename,
        ROOT_DIR / template_type / filename,
        Path.cwd() / "prompt" / question_dir / filename,
        Path.cwd() / question_dir / filename,
        Path.cwd() / template_type / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find question file for aspect='{aspect}', version='{question_version}', "
        f"template_type='{template_type}'. Tried: {[str(c) for c in candidates]}"
    )


def load_definition(question_data, template_type):
    definition = question_data.get("definition", "")
    if isinstance(definition, dict):
        if template_type in definition:
            return str(definition[template_type])
        if len(definition) == 1:
            return str(next(iter(definition.values())))
        return " / ".join(f"{k}: {v}" for k, v in definition.items())
    return str(definition)


def extract_all_questions(question_data):
    if "sub_aspect" in question_data:
        raw_sub_dimensions = question_data["sub_aspect"]
    elif "sub_dimension" in question_data:
        raw_sub_dimensions = question_data["sub_dimension"]
    else:
        raise ValueError("Question YAML must contain either 'sub_aspect' or 'sub_dimension'")

    if not isinstance(raw_sub_dimensions, dict):
        raise ValueError("Question sub-dimension payload must be a dictionary")

    all_questions = []
    for _, raw_questions in raw_sub_dimensions.items():
        all_questions.extend(_normalize_questions(raw_questions))
    return all_questions


def build_prompt(template_type, template, aspect, definition, question_list, row):
    prompt = template.replace("<aspect>", aspect).replace("<definition>", definition)
    prompt = prompt.replace("<questions>", question_list)

    if template_type == "summeval":
        prompt = prompt.replace("<source>", str(row.get("source", "")))
        prompt = prompt.replace("<summary>", str(row.get("system_output", row.get("summary", ""))))
    else:
        prompt = prompt.replace("<document>", str(row.get("document", row.get("source", ""))))
        prompt = prompt.replace("<fact>", str(row.get("fact", row.get("context", ""))))
        prompt = prompt.replace("<response>", str(row.get("response", row.get("system_output", ""))))
    return prompt


def main(
    data_path,
    base_url,
    model,
    aspect_list,
    question_version,
    save_dir,
    template_type,
    processor_type,
    temperature,
):
    api_key = os.getenv("OPENAI_API_KEY", "")
    processor = (
        vLLMProcessor(api_key=api_key or "EMPTY", base_url=base_url, model=model)
        if processor_type == "vllm"
        else OpenaiProcessor(api_key=api_key, model=model)
    )

    data = pd.read_csv(data_path)
    data = Dataset.from_pandas(data)
    os.makedirs(save_dir, exist_ok=True)

    template = summeval_template if template_type == "summeval" else topical_chat_template

    for aspect in tqdm(aspect_list):
        question_path = resolve_question_path(template_type, aspect, question_version)
        with open(question_path, "r", encoding="utf-8") as file:
            question_data = yaml.safe_load(file)

        all_questions = extract_all_questions(question_data)
        question_list = make_question_list(all_questions)
        definition = load_definition(question_data, template_type)

        def _mapping(row):
            return {
                "prompt": build_prompt(
                    template_type=template_type,
                    template=template,
                    aspect=aspect,
                    definition=definition,
                    question_list=question_list,
                    row=row,
                )
            }

        data_ds = data.map(_mapping)
        responses = asyncio.run(processor.process(data_ds["prompt"], temperature=temperature))
        data_ds = data_ds.add_column(
            f"{aspect}_response", [r.choices[0].message.content for r in responses]
        )
        data_ds.to_csv(os.path.join(save_dir, f"{aspect}_responses.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process some data using OpenaiProcessor or vLLMProcessor."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--aspects", nargs="+", required=True)
    parser.add_argument("--question_version", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--template_type", type=str, choices=["summeval", "topical_chat"], required=True
    )
    parser.add_argument("--processor_type", type=str, choices=["openai", "vllm"], required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    main(
        args.data_path,
        args.base_url,
        args.model,
        args.aspects,
        args.question_version,
        args.save_dir,
        args.template_type,
        args.processor_type,
        args.temperature,
    )
