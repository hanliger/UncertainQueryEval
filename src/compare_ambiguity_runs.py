import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Tuple


DETAILS_FILE = "ambiguity_eval_details.jsonl"
SUMMARY_FILE = "ambiguity_eval_summary.json"
MATRIX_FILE = "ambiguity_eval_matrix.csv"

MATRIX_SCORE_COLUMNS = [
    "user_utterance_scores",
    "dialogue_context_scores",
    "knowledge_base_scores",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def load_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / SUMMARY_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return load_json(path)


def load_details(run_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    path = run_dir / DETAILS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing details file: {path}")

    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (str(row["mode"]), str(row["scenario_id"]), str(row["dimension"]))
            out[key] = {
                "score": float(row["score"]),
                "taxonomy_code": row.get("taxonomy_code"),
                "title": row.get("title"),
            }
    return out


def load_matrix(run_dir: Path) -> Dict[Tuple[str, str], Dict[str, List[float]]]:
    path = run_dir / MATRIX_FILE
    if not path.exists():
        return {}

    out: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            key = (str(row.get("mode", "")), str(row.get("modality", "")))
            parsed: Dict[str, List[float]] = {}
            for col in MATRIX_SCORE_COLUMNS:
                raw = row.get(col, "[]")
                try:
                    arr = json.loads(raw)
                except json.JSONDecodeError:
                    arr = []
                parsed[col] = [float(v) for v in arr]
            out[key] = parsed
    return out


def _mode_summary_map(summary: Mapping[str, Any], mode: str, key: str) -> Dict[str, float]:
    items = summary.get("summary", {}).get(mode, {}).get(key, [])
    out: Dict[str, float] = {}
    for item in items:
        name = item.get("dimension") if key == "per_dimension" else item.get("scenario_id")
        if name is None:
            continue
        out[str(name)] = float(item.get("avg_score", 0.0))
    return out


def compare_summary(summary_a: Mapping[str, Any], summary_b: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"modes": {}}
    modes = sorted(
        set(summary_a.get("summary", {}).keys()) | set(summary_b.get("summary", {}).keys())
    )

    for mode in modes:
        mode_a = summary_a.get("summary", {}).get(mode, {})
        mode_b = summary_b.get("summary", {}).get(mode, {})
        overall_a = mode_a.get("overall_avg_score")
        overall_b = mode_b.get("overall_avg_score")

        per_dim_a = _mode_summary_map(summary_a, mode, "per_dimension")
        per_dim_b = _mode_summary_map(summary_b, mode, "per_dimension")
        per_scen_a = _mode_summary_map(summary_a, mode, "per_scenario")
        per_scen_b = _mode_summary_map(summary_b, mode, "per_scenario")

        dim_rows = []
        for dim in sorted(set(per_dim_a) | set(per_dim_b)):
            a = per_dim_a.get(dim)
            b = per_dim_b.get(dim)
            if a is None or b is None:
                continue
            dim_rows.append(
                {
                    "dimension": dim,
                    "a": a,
                    "b": b,
                    "diff_b_minus_a": b - a,
                    "abs_diff": abs(b - a),
                }
            )
        dim_rows.sort(key=lambda x: x["abs_diff"], reverse=True)

        scen_rows = []
        for sid in sorted(set(per_scen_a) | set(per_scen_b)):
            a = per_scen_a.get(sid)
            b = per_scen_b.get(sid)
            if a is None or b is None:
                continue
            scen_rows.append(
                {
                    "scenario_id": sid,
                    "a": a,
                    "b": b,
                    "diff_b_minus_a": b - a,
                    "abs_diff": abs(b - a),
                }
            )
        scen_rows.sort(key=lambda x: x["abs_diff"], reverse=True)

        result["modes"][mode] = {
            "overall": {
                "a": overall_a,
                "b": overall_b,
                "diff_b_minus_a": (overall_b - overall_a)
                if (overall_a is not None and overall_b is not None)
                else None,
            },
            "per_dimension": dim_rows,
            "per_scenario": scen_rows,
        }

    return result


def compare_details(
    details_a: Mapping[Tuple[str, str, str], Dict[str, Any]],
    details_b: Mapping[Tuple[str, str, str], Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    shared_keys = sorted(set(details_a.keys()) & set(details_b.keys()))
    if not shared_keys:
        return {
            "shared_record_count": 0,
            "mean_abs_diff": None,
            "median_abs_diff": None,
            "max_abs_diff": None,
            "per_mode": {},
            "top_changed_records": [],
        }

    abs_diffs: List[float] = []
    per_mode: Dict[str, List[float]] = {}
    changed_rows: List[Dict[str, Any]] = []

    for key in shared_keys:
        mode, scenario_id, dimension = key
        a = float(details_a[key]["score"])
        b = float(details_b[key]["score"])
        diff = b - a
        abs_diff = abs(diff)
        abs_diffs.append(abs_diff)
        per_mode.setdefault(mode, []).append(abs_diff)
        changed_rows.append(
            {
                "mode": mode,
                "scenario_id": scenario_id,
                "dimension": dimension,
                "a": a,
                "b": b,
                "diff_b_minus_a": diff,
                "abs_diff": abs_diff,
                "taxonomy_code": details_a[key].get("taxonomy_code"),
                "title": details_a[key].get("title"),
            }
        )

    changed_rows.sort(key=lambda row: row["abs_diff"], reverse=True)

    per_mode_stats = {}
    for mode, vals in per_mode.items():
        per_mode_stats[mode] = {
            "count": len(vals),
            "mean_abs_diff": mean(vals),
            "median_abs_diff": median(vals),
            "max_abs_diff": max(vals),
            "count_gt_0_05": sum(v > 0.05 for v in vals),
            "count_gt_0_10": sum(v > 0.10 for v in vals),
        }

    return {
        "shared_record_count": len(shared_keys),
        "mean_abs_diff": mean(abs_diffs),
        "median_abs_diff": median(abs_diffs),
        "max_abs_diff": max(abs_diffs),
        "count_gt_0_05": sum(v > 0.05 for v in abs_diffs),
        "count_gt_0_10": sum(v > 0.10 for v in abs_diffs),
        "per_mode": per_mode_stats,
        "top_changed_records": changed_rows[:top_k],
    }


def compare_matrix(
    matrix_a: Mapping[Tuple[str, str], Dict[str, List[float]]],
    matrix_b: Mapping[Tuple[str, str], Dict[str, List[float]]],
    top_k: int,
) -> Dict[str, Any]:
    shared_keys = sorted(set(matrix_a.keys()) & set(matrix_b.keys()))
    rows: List[Dict[str, Any]] = []

    for key in shared_keys:
        mode, modality = key
        a_row = matrix_a[key]
        b_row = matrix_b[key]
        for col in MATRIX_SCORE_COLUMNS:
            a_scores = a_row.get(col, [])
            b_scores = b_row.get(col, [])
            n = min(len(a_scores), len(b_scores))
            if n == 0:
                continue
            diffs = [b_scores[i] - a_scores[i] for i in range(n)]
            avg_diff = mean(diffs)
            rows.append(
                {
                    "mode": mode,
                    "modality": modality,
                    "element_scores_column": col,
                    "per_example_diff_b_minus_a": diffs,
                    "avg_diff_b_minus_a": avg_diff,
                    "abs_avg_diff": abs(avg_diff),
                    "max_abs_diff": max(abs(v) for v in diffs),
                }
            )

    rows.sort(key=lambda row: row["abs_avg_diff"], reverse=True)
    return {
        "shared_cell_count": len(rows),
        "top_changed_cells": rows[:top_k],
    }


def render_markdown(
    comparison: Mapping[str, Any],
    label_a: str,
    label_b: str,
    top_k: int,
) -> str:
    lines: List[str] = []
    lines.append(f"# Ambiguity Run Comparison ({label_a} vs {label_b})")
    lines.append("")

    summary_modes = comparison["summary_diff"]["modes"]
    lines.append("## Overall")
    for mode in sorted(summary_modes.keys()):
        overall = summary_modes[mode]["overall"]
        lines.append(
            f"- {mode}: a={overall['a']:.6f}, b={overall['b']:.6f}, "
            f"diff(b-a)={overall['diff_b_minus_a']:+.6f}"
        )
    lines.append("")

    details = comparison["record_level_diff"]
    lines.append("## Record Stability")
    lines.append(f"- shared_records: {details['shared_record_count']}")
    lines.append(f"- mean_abs_diff: {details['mean_abs_diff']:.6f}")
    lines.append(f"- max_abs_diff: {details['max_abs_diff']:.6f}")
    lines.append(f"- count_gt_0.05: {details['count_gt_0_05']}")
    lines.append(f"- count_gt_0.10: {details['count_gt_0_10']}")
    lines.append("")

    lines.append(f"## Top {top_k} Changed Records")
    lines.append("| mode | scenario_id | dimension | a | b | diff(b-a) | abs_diff |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in details["top_changed_records"]:
        lines.append(
            f"| {row['mode']} | {row['scenario_id']} | {row['dimension']} | "
            f"{row['a']:.4f} | {row['b']:.4f} | {row['diff_b_minus_a']:+.4f} | {row['abs_diff']:.4f} |"
        )
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two ambiguity evaluation run directories and save a diff report "
            "(comparison.json + comparison.md)."
        )
    )
    parser.add_argument("--run_a", type=str, required=True, help="Baseline run directory")
    parser.add_argument("--run_b", type=str, required=True, help="Compared run directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save outputs. Defaults to <run_b>/comparison_with_<run_a_name>.",
    )
    parser.add_argument("--output_name", type=str, default="comparison")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    if not run_a.exists():
        raise FileNotFoundError(f"run_a does not exist: {run_a}")
    if not run_b.exists():
        raise FileNotFoundError(f"run_b does not exist: {run_b}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_b / f"comparison_with_{run_a.name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_a = load_summary(run_a)
    summary_b = load_summary(run_b)
    details_a = load_details(run_a)
    details_b = load_details(run_b)
    matrix_a = load_matrix(run_a)
    matrix_b = load_matrix(run_b)

    comparison = {
        "config": {
            "run_a": str(run_a),
            "run_b": str(run_b),
            "label_a": run_a.name,
            "label_b": run_b.name,
            "top_k": args.top_k,
        },
        "summary_diff": compare_summary(summary_a, summary_b),
        "record_level_diff": compare_details(details_a, details_b, args.top_k),
        "matrix_diff": compare_matrix(matrix_a, matrix_b, args.top_k),
    }

    json_path = output_dir / f"{args.output_name}.json"
    md_path = output_dir / f"{args.output_name}.md"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(comparison, fp, ensure_ascii=False, indent=2)
    with md_path.open("w", encoding="utf-8") as fp:
        fp.write(render_markdown(comparison, run_a.name, run_b.name, args.top_k))

    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
