import argparse
import asyncio
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml
from openai import AsyncOpenAI, BadRequestError


ROOT_DIR = Path(__file__).resolve().parent.parent

ALL_DIMENSIONS = [
    "information_sufficiency",
    "information_coherence",
    "interpretation_singularity",
    "request_specificity",
    "temporal_determinacy",
    "utterance_completeness",
    "context_determinacy",
    "kb_alignment",
]

MODALITY_TO_DIMENSION = {
    "emptiness": "information_sufficiency",
    "결핍": "information_sufficiency",
    "clash": "information_coherence",
    "모순": "information_coherence",
    "multiplicity": "interpretation_singularity",
    "다중성": "interpretation_singularity",
    "generality": "request_specificity",
    "포괄성": "request_specificity",
    "fluidity": "temporal_determinacy",
    "유동성": "temporal_determinacy",
}

ELEMENT_TO_DIMENSION = {
    "user utterance": "utterance_completeness",
    "사용자 발화": "utterance_completeness",
    "dialogue context": "context_determinacy",
    "대화 문맥": "context_determinacy",
    "knowledge base": "kb_alignment",
    "지식 베이스": "kb_alignment",
}

MODALITY_MATRIX_ORDER = [
    ("Emptiness", "emptiness"),
    ("Clash", "clash"),
    ("Multiplicity", "multiplicity"),
    ("Generality", "generality"),
    ("Fluidity", "fluidity"),
]

ELEMENT_MATRIX_ORDER = [
    ("User Utterance", "user utterance"),
    ("Dialogue Context", "dialogue context"),
    ("Knowledge Base", "knowledge base"),
]


@dataclass
class SubDimensionQuestions:
    sub_dimension_name: str
    questions: List[str]


@dataclass
class DimensionQuestions:
    definition: str
    questions: List[str]
    sub_dimensions: Dict[str, List[str]]  # sub_dimension_name -> questions
    sub_dimensions: List[SubDimensionQuestions]


def _normalize_question(text: str) -> str:
    q = re.sub(r"\s+", " ", str(text)).strip()
    if not q:
        return ""
    if not q.endswith("?"):
        q = q.rstrip(".!") + "?"
    return q


def _extract_question_list(raw_sub_map: Mapping[str, Any]) -> List[str]:
    questions: List[str] = []
    for raw_value in raw_sub_map.values():
        if isinstance(raw_value, list):
            candidates = [str(v) for v in raw_value if str(v).strip()]
        else:
            text = str(raw_value or "").strip()
            if not text:
                candidates = []
            elif "?" in text:
                candidates = [chunk.strip() for chunk in re.split(r"\?\s*", text) if chunk.strip()]
            else:
                candidates = [line.strip("-• \t") for line in text.splitlines() if line.strip()]
        questions.extend([_normalize_question(c) for c in candidates if _normalize_question(c)])
    return questions


def load_dimension_questions(question_dir: Path, version: str) -> Dict[str, DimensionQuestions]:
    bundles: Dict[str, DimensionQuestions] = {}
    for dim in ALL_DIMENSIONS:
        path = question_dir / f"{dim}_{version}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Missing question file: {path}")
        with path.open("r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp)

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid YAML payload: {path}")

        definition_raw = payload.get("definition", "")
        if isinstance(definition_raw, dict):
            if "ambiguity" in definition_raw:
                definition = str(definition_raw["ambiguity"])
            elif len(definition_raw) == 1:
                definition = str(next(iter(definition_raw.values())))
            else:
                definition = " / ".join(f"{k}: {v}" for k, v in definition_raw.items())
        else:
            definition = str(definition_raw)

        raw_sub_map = payload.get("sub_dimension") or payload.get("sub_aspect")
        if not raw_sub_map or not isinstance(raw_sub_map, dict):
            raise ValueError(f"Missing or invalid sub-dimension key in: {path}")

        all_questions = _extract_question_list(raw_sub_map)
        sub_dims: Dict[str, List[str]] = {}
        for sub_name, raw_value in raw_sub_map.items():
            if isinstance(raw_value, list):
                candidates = [str(v) for v in raw_value if str(v).strip()]
            else:
                text = str(raw_value or "").strip()
                if not text:
                    candidates = []
                elif "?" in text:
                    candidates = [chunk.strip() for chunk in re.split(r"\?\s*", text) if chunk.strip()]
                else:
                    candidates = [line.strip("-• \t") for line in text.splitlines() if line.strip()]
            sub_qs = [_normalize_question(c) for c in candidates if _normalize_question(c)]
            if sub_qs:
                sub_dims[sub_name] = sub_qs

        if not all_questions:
            raise ValueError(f"No questions found in: {path}")

        bundles[dim] = DimensionQuestions(definition=definition, questions=all_questions, sub_dimensions=sub_dims)
    return bundles


def format_questions_for_prompt(questions: Sequence[str]) -> str:
    return "\n".join(f"Q{idx}: {q}" for idx, q in enumerate(questions, 1))


def format_conversation(conversation: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for turn in conversation:
        turn_no = turn.get("turn", "")
        role = turn.get("role", "unknown")
        content = str(turn.get("content", "")).strip()
        lines.append(f"Turn {turn_no} [{role}]: {content}")
    return "\n".join(lines)


def resolve_modality_dimension(modality_text: str) -> str:
    lowered = modality_text.lower()
    for key, dim in MODALITY_TO_DIMENSION.items():
        if key in lowered or key in modality_text:
            return dim
    raise ValueError(f"Unknown modality mapping: {modality_text}")


def resolve_element_dimension(element_text: str) -> str:
    lowered = element_text.lower()
    for key, dim in ELEMENT_TO_DIMENSION.items():
        if key in lowered or key in element_text:
            return dim
    raise ValueError(f"Unknown element mapping: {element_text}")


def dimensions_for_mode(scenario: Mapping[str, Any], mode: str) -> List[str]:
    if mode == "all":
        return list(ALL_DIMENSIONS)
    taxonomy = scenario.get("taxonomy", {})
    modality_dim = resolve_modality_dimension(str(taxonomy.get("modality", "")))
    element_dim = resolve_element_dimension(str(taxonomy.get("element", "")))
    dims = [modality_dim, element_dim]
    deduped = []
    for d in dims:
        if d not in deduped:
            deduped.append(d)
    return deduped


def build_ambiguity_prompt(
    scenario: Mapping[str, Any],
    dimension: str,
    definition: str,
    questions: Sequence[str],
) -> str:
    conversation = format_conversation(scenario.get("conversation", []))
    ambiguous_query = str(scenario.get("ambiguous_query", "")).strip()
    return f"""### Task Overview ###
Your task is to read a provided chatbot conversation and the final user query, then answer 'yes' or 'no' to specific questions.
These questions will relate to a particular dimension of the conversation's clarity.

### Dimension Definition ###
{dimension} - {definition}

### Instructions ###
1. Read these instructions thoroughly.
2. Carefully read the Conversation and the Final User Query.
3. Understand the given questions and the definition of the {dimension}.
4. Respond to each question with 'yes' or 'no'. Base your answers on a clear rationale.
5. Follow the specified format for your answers.

# Conversation #
{conversation}

# Final User Query #
"{ambiguous_query}"

### Answer Format ###
Q1: [Your Answer]
Q2: [Your Answer]
...

# Questions #
{format_questions_for_prompt(questions)}

# Response #
Provide your answers to the given questions, following the specified Answer Format.
"""


def parse_yes_no_answers(response_text: str, expected_count: int) -> List[int]:
    strong_matches = re.findall(r"Q\d+\s*:\s*(Yes|No)\b", response_text, flags=re.IGNORECASE)
    if strong_matches:
        picked = strong_matches[:expected_count]
        return [1 if m.lower() == "yes" else 0 for m in picked]
    fallback = re.findall(r"\b(Yes|No)\b", response_text, flags=re.IGNORECASE)
    picked = fallback[:expected_count]
    return [1 if m.lower() == "yes" else 0 for m in picked]


async def request_chat_completion(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 500,
    max_retries: int = 4,
) -> str:
    req = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            try:
                response = await client.chat.completions.create(
                    **req,
                    max_completion_tokens=max_tokens,
                )
            except BadRequestError as exc:
                if "max_completion_tokens" not in str(exc):
                    raise
                response = await client.chat.completions.create(
                    **req,
                    max_tokens=max_tokens,
                )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt == max_retries - 1:
                break
            await asyncio.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Failed after retries: {last_err}")


async def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    client: AsyncOpenAI,
    model: str,
    max_concurrency: int,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max_concurrency)

    async def _run_one(record: Mapping[str, Any]) -> Dict[str, Any]:
        async with sem:
            response_text = await request_chat_completion(
                client=client,
                model=model,
                prompt=record["prompt"],
                max_tokens=500,
            )
            answers = parse_yes_no_answers(response_text, record["expected_questions"])
            score = mean(answers) if answers else None
            return {
                **record,
                "response_text": response_text,
                "answers": answers,
                "answered_count": len(answers),
                "yes_count": sum(answers),
                "score": score,
                "parse_ok": len(answers) > 0,
            }

    tasks = [asyncio.create_task(_run_one(record)) for record in records]
    return await asyncio.gather(*tasks)


def aggregate_mode_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if r.get("score") is not None]

    by_dim = defaultdict(list)
    by_scenario = defaultdict(list)
    for row in valid:
        by_dim[row["dimension"]].append(float(row["score"]))
        by_scenario[row["scenario_id"]].append(float(row["score"]))

    per_dimension = []
    for dim in sorted(by_dim):
        values = by_dim[dim]
        per_dimension.append(
            {
                "dimension": dim,
                "count": len(values),
                "avg_score": mean(values) if values else None,
            }
        )

    per_scenario = []
    for scenario_id in sorted(by_scenario):
        values = by_scenario[scenario_id]
        per_scenario.append(
            {
                "scenario_id": scenario_id,
                "count": len(values),
                "avg_score": mean(values) if values else None,
            }
        )

    overall_avg = mean([float(r["score"]) for r in valid]) if valid else None
    return {
        "overall_avg_score": overall_avg,
        "record_count": len(results),
        "valid_record_count": len(valid),
        "per_dimension": per_dimension,
        "per_scenario": per_scenario,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "mode",
        "scenario_id",
        "dimension",
        "taxonomy_code",
        "taxonomy_modality",
        "taxonomy_element",
        "expected_questions",
        "answered_count",
        "yes_count",
        "score",
        "parse_ok",
        "title",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _canonical_label(raw_label: str, ordered: Sequence[tuple[str, str]]) -> str | None:
    lowered = raw_label.lower()
    for display_name, english_key in ordered:
        if english_key in lowered:
            return display_name
    return None


def _build_mode_matrix_rows(
    all_rows: Sequence[Mapping[str, Any]],
    mode: str,
) -> List[Dict[str, Any]]:
    by_scenario: Dict[Any, Dict[str, Any]] = defaultdict(
        lambda: {"scores": [], "modality": "", "element": ""}
    )
    for row in all_rows:
        if row.get("mode") != mode or row.get("score") is None:
            continue
        scenario_id = row.get("scenario_id")
        record = by_scenario[scenario_id]
        record["scores"].append(float(row["score"]))
        record["modality"] = str(row.get("taxonomy_modality", ""))
        record["element"] = str(row.get("taxonomy_element", ""))

    cell_values: Dict[tuple[str, str], List[tuple[Any, float]]] = defaultdict(list)
    for scenario_id, data in by_scenario.items():
        if not data["scores"]:
            continue
        modality_name = _canonical_label(data["modality"], MODALITY_MATRIX_ORDER)
        element_name = _canonical_label(data["element"], ELEMENT_MATRIX_ORDER)
        if not modality_name or not element_name:
            continue
        avg_score = mean(data["scores"])
        cell_values[(modality_name, element_name)].append((scenario_id, avg_score))

    for key in cell_values:
        cell_values[key].sort(key=lambda item: str(item[0]))

    rows: List[Dict[str, Any]] = []
    for modality_name, _ in MODALITY_MATRIX_ORDER:
        out = {"mode": mode, "modality": modality_name}
        for element_name, _ in ELEMENT_MATRIX_ORDER:
            entries = cell_values.get((modality_name, element_name), [])
            score_list = [round(score, 4) for _, score in entries]
            scenario_ids = [str(sid) for sid, _ in entries]
            col_prefix = element_name.lower().replace(" ", "_")
            out[f"{col_prefix}_scores"] = json.dumps(score_list, ensure_ascii=False)
            out[f"{col_prefix}_scenario_ids"] = json.dumps(scenario_ids, ensure_ascii=False)
        rows.append(out)
    return rows


def write_matrix_csv(path: Path, all_rows: Sequence[Mapping[str, Any]]) -> None:
    mode_order = ["matched", "all"]
    present_modes = {str(row.get("mode")) for row in all_rows}
    matrix_rows: List[Dict[str, Any]] = []
    for mode in mode_order:
        if mode in present_modes:
            matrix_rows.extend(_build_mode_matrix_rows(all_rows, mode))

    if not matrix_rows:
        return

    fieldnames = [
        "mode",
        "modality",
        "user_utterance_scores",
        "dialogue_context_scores",
        "knowledge_base_scores",
        "user_utterance_scenario_ids",
        "dialogue_context_scenario_ids",
        "knowledge_base_scenario_ids",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in matrix_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_matrix_markdown(path: Path, all_rows: Sequence[Mapping[str, Any]]) -> None:
    mode_order = ["matched", "all"]
    present_modes = {str(row.get("mode")) for row in all_rows}
    lines: List[str] = []

    for mode in mode_order:
        if mode not in present_modes:
            continue
        matrix_rows = _build_mode_matrix_rows(all_rows, mode)
        lines.append(f"## {mode.upper()}")
        lines.append(
            "| Modality \\\\ Element | User Utterance | Dialogue Context | Knowledge Base |"
        )
        lines.append("|---|---|---|---|")
        for row in matrix_rows:
            lines.append(
                "| {modality} | {uu} | {dc} | {kb} |".format(
                    modality=row["modality"],
                    uu=row["user_utterance_scores"],
                    dc=row["dialogue_context_scores"],
                    kb=row["knowledge_base_scores"],
                )
            )
        lines.append("")

    if not lines:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines).strip() + "\n")


def resolve_run_save_dir(base_dir: Path, run_name: str) -> Path:
    raw = run_name.strip()
    if raw:
        candidate = base_dir / raw
    else:
        # Include microseconds to avoid collisions when runs start close together.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        candidate = base_dir / stamp

    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        alt = candidate.parent / f"{candidate.name}_{suffix}"
        if not alt.exists():
            return alt
        suffix += 1


async def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    question_dir = Path(args.question_dir)
    save_root_dir = Path(args.save_dir)
    save_dir = resolve_run_save_dir(save_root_dir, args.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    with data_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    scenarios = payload.get("scenarios", [])
    if not scenarios:
        raise ValueError(f"No scenarios found in: {data_path}")

    question_bundles = load_dimension_questions(question_dir, args.question_version)
    modes = ["matched", "all"] if args.mode == "both" else [args.mode]

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
        client_kwargs = {"api_key": api_key or "EMPTY", "base_url": args.base_url}

    async with AsyncOpenAI(**client_kwargs) as client:
        all_rows: List[Dict[str, Any]] = []
        mode_summaries: Dict[str, Any] = {}

        for mode in modes:
            eval_records: List[Dict[str, Any]] = []
            for scenario in scenarios:
                dims = dimensions_for_mode(scenario, mode)
                for dim in dims:
                    bundle = question_bundles[dim]
                    taxonomy = scenario.get("taxonomy", {})
                    for sub_name, sub_questions in bundle.sub_dimensions.items():
                        prompt = build_ambiguity_prompt(
                            scenario=scenario,
                            dimension=dim,
                            definition=bundle.definition,
                            questions=sub_questions,
                        )
                        eval_records.append(
                            {
                                "mode": mode,
                                "scenario_id": scenario.get("scenario_id"),
                                "dimension": dim,
                                "sub_dimension": sub_name,
                                "taxonomy_code": taxonomy.get("code"),
                                "taxonomy_modality": taxonomy.get("modality"),
                                "taxonomy_element": taxonomy.get("element"),
                                "title": scenario.get("title"),
                                "expected_questions": len(sub_questions),
                                "prompt": prompt,
                            }
                        )

            print(f"[run] mode={mode} records={len(eval_records)}")
            mode_rows = await evaluate_records(
                records=eval_records,
                client=client,
                model=args.model,
                max_concurrency=args.max_concurrency,
            )
            all_rows.extend(mode_rows)
            mode_summaries[mode] = aggregate_mode_results(mode_rows)
            print(
                f"[done] mode={mode} "
                f"overall_avg_score={mode_summaries[mode]['overall_avg_score']}"
            )

    write_csv(save_dir / "ambiguity_eval_details.csv", all_rows)
    with (save_dir / "ambiguity_eval_details.jsonl").open("w", encoding="utf-8") as fp:
        for row in all_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "config": {
            "data_path": str(data_path),
            "question_dir": str(question_dir),
            "question_version": args.question_version,
            "mode": args.mode,
            "backend": args.backend,
            "model": args.model,
            "max_concurrency": args.max_concurrency,
            "save_dir_base": str(save_root_dir),
            "save_dir_resolved": str(save_dir),
            "run_name": args.run_name,
        },
        "summary": mode_summaries,
    }
    with (save_dir / "ambiguity_eval_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    matrix_csv_path = save_dir / "ambiguity_eval_matrix.csv"
    matrix_md_path = save_dir / "ambiguity_eval_matrix.md"
    write_matrix_csv(matrix_csv_path, all_rows)
    write_matrix_markdown(matrix_md_path, all_rows)

    print(f"[saved] {save_dir / 'ambiguity_eval_details.csv'}")
    print(f"[saved] {save_dir / 'ambiguity_eval_details.jsonl'}")
    print(f"[saved] {save_dir / 'ambiguity_eval_summary.json'}")
    print(f"[saved] {matrix_csv_path}")
    print(f"[saved] {matrix_md_path}")
    print(f"[done] run_dir={save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ambiguity scenarios with CheckEval-style checklist prompts. "
            "Supports matched (taxonomy-mapped dimensions) and all-dimensions modes."
        )
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(ROOT_DIR / "data" / "ambigous_query" / "ambiguity_scenarios.json"),
    )
    parser.add_argument(
        "--question_dir",
        type=str,
        default=str(ROOT_DIR / "prompt" / "ambiguity_questions"),
    )
    parser.add_argument("--question_version", type=str, default="filtered")
    parser.add_argument("--mode", choices=["matched", "all", "both"], default="both")
    parser.add_argument("--backend", choices=["openai", "vllm"], default="openai")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_concurrency", type=int, default=8)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(ROOT_DIR / "results" / "ambiguity_checkeval"),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help=(
            "Optional subdirectory name under --save_dir. "
            "If omitted, a timestamped run directory is created automatically."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
