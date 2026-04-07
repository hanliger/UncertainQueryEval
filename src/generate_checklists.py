import argparse
import asyncio
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml
from openai import AsyncOpenAI, BadRequestError


BENCHMARK_DESCRIPTIONS = {
    "summeval": "Evaluation of summary quality for news summarization outputs.",
    "topical_chat": "Evaluation of knowledge-grounded dialogue response quality.",
    "ambiguity": "Evaluation of ambiguity in multi-turn financial assistant queries.",
}

EN_WH_WORDS = {
    "what",
    "why",
    "how",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
}
KO_WH_PREFIXES = ("무엇", "왜", "어떻게", "어떤", "누구", "언제", "어디")
QUESTION_KEYS = ("sub_aspect", "sub_dimension")


@dataclass
class DimensionChecklist:
    name: str
    definition: Any
    question_key: str
    sub_questions: "OrderedDict[str, List[str]]"


def _slugify_dimension_name(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
    slug = slug.strip("_").lower()
    return slug or "dimension"


def _dedupe_preserving_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _ensure_question_mark(text: str) -> str:
    text = _normalize_whitespace(text)
    if not text:
        return ""
    if text.endswith("?"):
        return text
    text = text.rstrip(".!")
    return f"{text}?"


def _looks_like_yes_no_question(question: str) -> bool:
    q = question.strip()
    if not q.endswith("?"):
        return False
    lowered = q.lower()
    first_word_match = re.match(r"[a-zA-Z']+", lowered)
    if first_word_match and first_word_match.group(0) in EN_WH_WORDS:
        return False
    if any(q.startswith(prefix) for prefix in KO_WH_PREFIXES):
        return False
    return True


def _canonical_question(question: str) -> str:
    return _normalize_whitespace(_ensure_question_mark(question)).lower()


def _split_questions(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        candidates = [str(v) for v in raw_value if str(v).strip()]
    else:
        text = str(raw_value).strip()
        if not text:
            return []
        if "?" in text:
            candidates = [chunk.strip() for chunk in re.split(r"\?\s*", text) if chunk.strip()]
        else:
            candidates = [line.strip("-• \t") for line in text.splitlines() if line.strip()]
    normalized = []
    for candidate in candidates:
        question = _ensure_question_mark(candidate)
        if not question:
            continue
        normalized.append(question)
    return _dedupe_preserving_order(normalized)


def _coerce_definition(raw_definition: Any) -> Any:
    if isinstance(raw_definition, Mapping):
        return {str(k): _normalize_whitespace(str(v)) for k, v in raw_definition.items()}
    return _normalize_whitespace(str(raw_definition))


def _extract_question_key(payload: Mapping[str, Any]) -> str:
    for key in QUESTION_KEYS:
        if key in payload:
            return key
    raise ValueError("No question key found. Expected one of: sub_aspect, sub_dimension")


def _normalize_sub_questions(raw_sub_questions: Mapping[str, Any]) -> "OrderedDict[str, List[str]]":
    ordered = OrderedDict()
    for sub_name, raw_questions in raw_sub_questions.items():
        ordered[str(sub_name)] = _split_questions(raw_questions)
    return ordered


def _build_dimension_from_payload(name: str, payload: Mapping[str, Any]) -> DimensionChecklist:
    if "definition" not in payload:
        raise ValueError(f"Dimension '{name}' is missing 'definition'")
    question_key = _extract_question_key(payload)
    raw_sub_questions = payload.get(question_key, {})
    if not isinstance(raw_sub_questions, Mapping):
        raise ValueError(
            f"Dimension '{name}' has invalid '{question_key}'. Expected a dictionary."
        )
    sub_questions = _normalize_sub_questions(raw_sub_questions)
    if not sub_questions:
        raise ValueError(f"Dimension '{name}' has no sub-dimensions/questions")
    return DimensionChecklist(
        name=name,
        definition=_coerce_definition(payload["definition"]),
        question_key=question_key,
        sub_questions=sub_questions,
    )


def parse_seed_dimensions(seed_path: Path) -> List[DimensionChecklist]:
    with seed_path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Seed file is not a dictionary: {seed_path}")

    if "definition" in payload and any(key in payload for key in QUESTION_KEYS):
        dimension_name = seed_path.stem.replace("_seed", "")
        return [_build_dimension_from_payload(dimension_name, payload)]

    dimensions = []
    for name, dim_payload in payload.items():
        if not isinstance(dim_payload, Mapping):
            continue
        if "definition" in dim_payload and any(k in dim_payload for k in QUESTION_KEYS):
            dimensions.append(_build_dimension_from_payload(str(name), dim_payload))

    if not dimensions:
        raise ValueError(
            f"Could not detect dimension payload in {seed_path}. "
            "Expected single-dimension or multi-dimension seed format."
        )
    return dimensions


def collect_seed_inputs(seed_input: Path) -> List[Path]:
    if seed_input.is_file():
        return [seed_input]
    if seed_input.is_dir():
        files = sorted(seed_input.glob("*_seed.yaml"))
        if not files:
            raise ValueError(f"No '*_seed.yaml' files found in directory: {seed_input}")
        return files
    raise ValueError(f"seed_input does not exist: {seed_input}")


def definition_to_text(definition: Any, benchmark_name: str) -> str:
    if isinstance(definition, Mapping):
        if benchmark_name in definition:
            return str(definition[benchmark_name])
        if len(definition) == 1:
            return str(next(iter(definition.values())))
        return " / ".join(f"{k}: {v}" for k, v in definition.items())
    return str(definition)


def format_sub_questions_for_prompt(sub_questions: Mapping[str, Sequence[str]]) -> str:
    return json.dumps(sub_questions, ensure_ascii=False, indent=2)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    stripped = strip_code_fence(text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model output does not contain a JSON object")
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be an object")
    return parsed


def normalize_generated_questions(
    parsed_output: Mapping[str, Any], allowed_sub_dimensions: Sequence[str]
) -> "OrderedDict[str, List[str]]":
    normalized = OrderedDict()
    for sub_dim in allowed_sub_dimensions:
        raw_questions = parsed_output.get(sub_dim, [])
        if isinstance(raw_questions, str):
            questions = _split_questions(raw_questions)
        elif isinstance(raw_questions, list):
            questions = []
            for item in raw_questions:
                if not isinstance(item, str):
                    continue
                question = _ensure_question_mark(item)
                if not question:
                    continue
                questions.append(question)
        else:
            questions = []
        validated = []
        for question in questions:
            if not _looks_like_yes_no_question(question):
                continue
            validated.append(question)
        normalized[sub_dim] = _dedupe_preserving_order(validated)
    return normalized


def merge_seed_and_generated(
    seed_questions: Mapping[str, Sequence[str]],
    generated_questions: Mapping[str, Sequence[str]],
) -> "OrderedDict[str, List[str]]":
    merged = OrderedDict()
    for sub_dim, seed_list in seed_questions.items():
        merged[sub_dim] = _dedupe_preserving_order(
            list(seed_list) + list(generated_questions.get(sub_dim, []))
        )
    return merged


def build_combined_pool(
    seed_questions: Mapping[str, Sequence[str]],
    diversification_generated: Mapping[str, Sequence[str]],
    elaboration_generated: Mapping[str, Sequence[str]],
) -> "OrderedDict[str, List[str]]":
    combined = OrderedDict()
    for sub_dim, seed_list in seed_questions.items():
        combined[sub_dim] = _dedupe_preserving_order(
            list(seed_list)
            + list(diversification_generated.get(sub_dim, []))
            + list(elaboration_generated.get(sub_dim, []))
        )
    return combined


def normalize_filtered_questions(
    filtered_output: Mapping[str, Any], combined_pool: Mapping[str, Sequence[str]]
) -> "OrderedDict[str, List[str]]":
    normalized = OrderedDict()
    for sub_dim, source_questions in combined_pool.items():
        source_map = {
            _canonical_question(question): question for question in source_questions
        }
        raw_questions = filtered_output.get(sub_dim, [])
        if isinstance(raw_questions, str):
            candidates = _split_questions(raw_questions)
        elif isinstance(raw_questions, list):
            candidates = [str(item) for item in raw_questions if isinstance(item, str)]
        else:
            candidates = []

        kept = []
        for candidate in candidates:
            normalized_candidate = _canonical_question(candidate)
            if normalized_candidate in source_map:
                original_question = source_map[normalized_candidate]
                if _looks_like_yes_no_question(original_question):
                    kept.append(original_question)
        normalized[sub_dim] = _dedupe_preserving_order(kept)
    return normalized


def build_diversification_prompt(
    benchmark_name: str,
    benchmark_description: str,
    dimension_name: str,
    definition_text: str,
    seed_sub_questions: Mapping[str, Sequence[str]],
) -> str:
    return f"""<Task Overview>
You will be provided with:
1) Information about the benchmark
2) The main concept being assessed
3) Seed questions grouped by sub-dimensions
Your task is to create additional sub-questions to expand viewpoint diversity while preserving the same dimension intent.

1) Benchmark Information:
- Benchmark Name: {benchmark_name}
- Benchmark Description: {benchmark_description}

2) Main Concept:
- Dimension: {dimension_name}
- Definition: {definition_text}

3) Sub-dimensions and Seed Questions (JSON):
{format_sub_questions_for_prompt(seed_sub_questions)}

<Conditions for a Good Question List>
- Each question must be answerable with yes/no.
- A yes answer must indicate better quality on this dimension.
- Each question must target one concept only.
- Avoid overlap with existing seed questions and avoid near-duplicates.
- Keep language aligned with the seed language.

<Output Requirements>
- Return JSON only (no markdown, no explanation).
- Keep the original sub-dimension keys.
- Output only additional questions.
- JSON schema:
{{
  "Sub-dimension A": ["New question 1?", "New question 2?"],
  "Sub-dimension B": ["New question 3?"]
}}
"""


def build_elaboration_prompt(
    benchmark_name: str,
    benchmark_description: str,
    dimension_name: str,
    definition_text: str,
    seed_sub_questions: Mapping[str, Sequence[str]],
) -> str:
    return f"""<Task Overview>
Generate additional yes/no questions that elaborate seed questions into more specific and fine-grained checks.
Diversification and elaboration are independent processes: use only the seed questions below as source.

1) Benchmark Information:
- Benchmark Name: {benchmark_name}
- Benchmark Description: {benchmark_description}

2) Target Dimension:
- Dimension: {dimension_name}
- Definition: {definition_text}

3) Sub-dimensions and Seed Questions (JSON):
{format_sub_questions_for_prompt(seed_sub_questions)}

<Conditions for a Good Question List>
- Each question must be answerable with yes/no.
- A yes answer must indicate better quality on this dimension.
- Questions should be more specific than seed questions, not broader.
- Each question should focus on one criterion only.
- Do not create questions for unrelated dimensions.
- Keep language aligned with the seed language.

<Output Requirements>
- Return JSON only.
- Keep original sub-dimension keys.
- Output only additional elaborated questions.
- JSON schema:
{{
  "Sub-dimension A": ["Detailed question 1?", "Detailed question 2?"],
  "Sub-dimension B": ["Detailed question 3?"]
}}
"""


def build_filtering_prompt(
    benchmark_name: str,
    benchmark_description: str,
    dimension_name: str,
    definition_text: str,
    combined_pool: Mapping[str, Sequence[str]],
) -> str:
    return f"""<Task Overview>
Filter questions by removing only low-quality entries from the combined pool.

1) Dimension Alignment
- Dimension: {dimension_name}
- Definition: {definition_text}
- Remove questions that do not align with the dimension definition.
- Remove questions where a "yes" does not imply better quality for this dimension.

2) Redundancy
- Remove semantically overlapping questions, even if wording differs.

3) Style
- Remove overly exaggerated wording.
- Remove excessively narrow/minor questions that do not meaningfully affect overall quality.

4) Benchmark Context
- Name: {benchmark_name}
- Description: {benchmark_description}

5) Sub-dimensions and Questions (JSON):
{format_sub_questions_for_prompt(combined_pool)}

<Output Requirements>
- Return JSON only.
- Keep original sub-dimension keys.
- Do not rewrite any retained question text.
- Do not generate new questions.
- Remove only.
- JSON schema:
{{
  "Sub-dimension A": ["Retained question 1?", "Retained question 2?"],
  "Sub-dimension B": []
}}
"""


async def request_model_json(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 4000,
) -> Dict[str, Any]:
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    try:
        # Newer models (for example gpt-5.*) require max_completion_tokens.
        response = await client.chat.completions.create(
            **request_kwargs,
            max_completion_tokens=max_tokens,
        )
    except BadRequestError as exc:
        # Backward compatibility for models/endpoints that still expect max_tokens.
        if "max_completion_tokens" not in str(exc):
            raise
        response = await client.chat.completions.create(
            **request_kwargs,
            max_tokens=max_tokens,
        )
    content = response.choices[0].message.content or ""
    return parse_json_object(content)


def dump_dimension_file(
    output_path: Path,
    definition: Any,
    question_key: str,
    sub_questions: Mapping[str, Sequence[str]],
) -> None:
    payload: Dict[str, Any] = OrderedDict()
    payload["definition"] = definition
    payload[question_key] = OrderedDict(
        (sub_dim, list(questions)) for sub_dim, questions in sub_questions.items()
    )

    def _to_builtin(value: Any) -> Any:
        if isinstance(value, OrderedDict):
            return {k: _to_builtin(v) for k, v in value.items()}
        if isinstance(value, dict):
            return {k: _to_builtin(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_to_builtin(v) for v in value]
        return value

    dump_payload = _to_builtin(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(dump_payload, fp, allow_unicode=True, sort_keys=False, width=120)


async def run_pipeline(args: argparse.Namespace) -> None:
    seed_inputs = collect_seed_inputs(Path(args.seed_input))
    output_dir = Path(args.output_dir)
    benchmark_name = args.benchmark_name
    benchmark_description = BENCHMARK_DESCRIPTIONS.get(
        benchmark_name.lower(), f"Evaluation benchmark for {benchmark_name}"
    )

    all_dimensions: List[DimensionChecklist] = []
    for seed_file in seed_inputs:
        dimensions = parse_seed_dimensions(seed_file)
        all_dimensions.extend(dimensions)

    slug_to_dimension = {}
    for dim in all_dimensions:
        slug = _slugify_dimension_name(dim.name)
        if slug in slug_to_dimension and slug_to_dimension[slug] != dim.name:
            raise ValueError(
                f"Dimension name collision after slugify: '{slug_to_dimension[slug]}' and '{dim.name}'"
            )
        slug_to_dimension[slug] = dim.name

    api_key = os.getenv(args.api_key_env, "")
    if args.backend == "openai":
        if not api_key:
            raise ValueError(
                f"Missing API key. Set env var '{args.api_key_env}' for OpenAI backend."
            )
        client_kwargs = {"api_key": api_key}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
    else:
        if not args.base_url:
            raise ValueError("vLLM backend requires --base_url")
        client_kwargs = {
            "api_key": api_key or "EMPTY",
            "base_url": args.base_url,
        }

    async with AsyncOpenAI(**client_kwargs) as client:
        for dim in all_dimensions:
            dim_slug = _slugify_dimension_name(dim.name)
            definition_text = definition_to_text(dim.definition, benchmark_name)
            sub_dimension_names = list(dim.sub_questions.keys())

            diversification_prompt = build_diversification_prompt(
                benchmark_name=benchmark_name,
                benchmark_description=benchmark_description,
                dimension_name=dim.name,
                definition_text=definition_text,
                seed_sub_questions=dim.sub_questions,
            )
            elaboration_prompt = build_elaboration_prompt(
                benchmark_name=benchmark_name,
                benchmark_description=benchmark_description,
                dimension_name=dim.name,
                definition_text=definition_text,
                seed_sub_questions=dim.sub_questions,
            )

            diversification_raw = await request_model_json(
                client=client, model=args.model, prompt=diversification_prompt
            )
            elaboration_raw = await request_model_json(
                client=client, model=args.model, prompt=elaboration_prompt
            )

            diversification_generated = normalize_generated_questions(
                diversification_raw, sub_dimension_names
            )
            elaboration_generated = normalize_generated_questions(
                elaboration_raw, sub_dimension_names
            )

            diversification_full = merge_seed_and_generated(
                dim.sub_questions, diversification_generated
            )
            elaboration_full = merge_seed_and_generated(
                dim.sub_questions, elaboration_generated
            )

            combined_pool = build_combined_pool(
                dim.sub_questions,
                diversification_generated,
                elaboration_generated,
            )

            filtering_prompt = build_filtering_prompt(
                benchmark_name=benchmark_name,
                benchmark_description=benchmark_description,
                dimension_name=dim.name,
                definition_text=definition_text,
                combined_pool=combined_pool,
            )

            filtered_raw = await request_model_json(
                client=client, model=args.model, prompt=filtering_prompt
            )
            filtered_questions = normalize_filtered_questions(filtered_raw, combined_pool)

            dump_dimension_file(
                output_dir / f"{dim_slug}_seed.yaml",
                dim.definition,
                dim.question_key,
                dim.sub_questions,
            )
            dump_dimension_file(
                output_dir / f"{dim_slug}_diversification.yaml",
                dim.definition,
                dim.question_key,
                diversification_full,
            )
            dump_dimension_file(
                output_dir / f"{dim_slug}_elaboration.yaml",
                dim.definition,
                dim.question_key,
                elaboration_full,
            )
            dump_dimension_file(
                output_dir / f"{dim_slug}_filtered.yaml",
                dim.definition,
                dim.question_key,
                filtered_questions,
            )

            print(f"[done] {dim.name} -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate diversification/elaboration/filtered checklists from seed questions."
    )
    parser.add_argument("--seed_input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--benchmark_name", type=str, required=True)
    parser.add_argument("--backend", type=str, choices=["openai", "vllm"], default="openai")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    args = parser.parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
