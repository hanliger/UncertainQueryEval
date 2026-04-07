# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UncertainQueryEval is the project name for this repository. It contains the original CheckEval-style evaluation pipeline for reliable LLM-as-a-Judge scoring with checklists, along with the ambiguity evaluation extensions used for uncertain user queries. Paper: arXiv:2403.18771.

## Running Inference

There is no build system, package manager config, or requirements.txt. Dependencies must be installed manually: `openai`, `datasets`, `pandas`, `numpy`, `scipy`, `pyyaml`, `tqdm`, `nest_asyncio`, `prettytable`.

### CheckEval inference

```bash
python src/inference_checkeval.py \
  --data_path ./data/summeval/summeval_result.csv \
  --base_url <VLLM_URL> \
  --model <MODEL_NAME> \
  --save_dir ./results \
  --question_version seed \
  --aspects coherence consistency fluency relevance \
  --template_type summeval \
  --processor_type vllm
```

### G-Eval baseline inference

```bash
python src/inference_geval.py \
  --base_url <VLLM_URL> \
  --model <MODEL_NAME> \
  --save_dir ./results
```

### Batch orchestration

`src/vllm_inference.sh` runs CheckEval and G-Eval across both datasets (summeval, topical_chat) and all three question versions (seed, diversification, elaboration). Edit `BASE_URL` and `MODEL` before running.

## End-to-End Pipeline Flow

The full CheckEval pipeline has 5 stages. Stages 1-2 happen outside this codebase; Stages 3-5 are implemented in `src/`.

### Stage 1: Defining Dimensions of Evaluation (manual, outside code)

Humans define evaluation **dimensions** and **sub-dimensions** for a given task. Results are stored as YAML files in `prompt/`.

- **SummEval**: coherence, consistency, fluency, relevance
- **Topical-Chat**: naturalness, coherence, engagingness, groundedness

Each YAML contains `definition` (dimension definition) and `sub_aspect` (Boolean questions per sub-dimension).

### Stage 2: Checklist Generation (outside code, done with GPT-4o)

Seed questions are augmented by LLM. This process is NOT implemented in this repo -- only the resulting YAML files are checked in.

1. **Seed** -- human-written original questions (`*_seed.yaml`)
2. **Diversification** -- same question rephrased from different perspectives (`*_diversification.yaml`)
3. **Elaboration** -- more specific and detailed questions (`*_elaboration.yaml`)
4. **Filtering** -- LLM-based filtering by alignment, dimension consistency, redundancy removal (also not in code)

### Stage 3: Checklist-based Evaluation (`inference_checkeval.py`)

Core inference loop:

1. `main()` loads CSV data and iterates over each aspect
2. For each aspect, loads YAML and calls `make_question_list()` to format questions as `Q1: ...?\nQ2: ...?`
3. Fills `summeval_template` or `topical_chat_template` by replacing `<aspect>`, `<definition>`, `<source>`, `<summary>`, `<questions>` etc.
4. `vLLMProcessor.process()` or `OpenaiProcessor.process()` sends prompts via AsyncOpenAI in parallel
5. LLM responds in `Q1: Yes\nQ2: No\n...` format; saved as `{aspect}_response` column to CSV

Inference params from paper (Section 4.4): `temperature=0`, `n=1`, `max_length=200`.

### Stage 4: Score Aggregation (`aggregation.py`)

```
LLM response text -> parse_output() extracts Yes/No -> [1, 0, 1, 1, ...]
  -> Aggregator.calculate_metrics() computes mean, proportion, std_dev, etc.
```

Final score = proportion of "Yes" answers (e.g., 15 Yes out of 20 questions = 0.75). Uniform weighting across all questions.

### Stage 5: Correlation Analysis (`correlation.py`)

Computes Pearson, Spearman, Kendall correlations between predicted scores and human judgments at three levels:

- **Sample level**: per individual sample
- **Summary level**: averaged per document (`doc_id`), then correlated
- **System level**: averaged per model (`system_id`), then correlated

### Not Implemented in This Repo

The following are described in the paper but not present in code:

- **Question Augmentation** (Diversification/Elaboration generation) -- done externally with GPT-4o, results stored as YAML
- **Question Filtering** -- alignment, dimension consistency, redundancy removal
- **SEEval baseline** -- only G-Eval is implemented as a comparison baseline
- **IEA (Inter-Evaluator Agreement)** -- Krippendorff's alpha, Fleiss' kappa calculations

## Prompt Templates

YAML files in `prompt/` define evaluation checklists. Each template type has three question versions:
- **seed**: Original minimal questions
- **diversification**: Multiple phrasings of the same question
- **elaboration**: More detailed, complex formulations

Templates are organized by dataset (`topical_chat_questions/`, `summeval_questions/`) and by aspect.

## Datasets

- **SummEval** (`data/summeval/`): News article summarization evaluation. Key fields: `doc_id`, `system_id`, `source`, `system_output`, `scores`.
- **Topical Chat** (`data/topical_chat/`): Conversational response evaluation. Key fields: `document` (history), `fact`, `response`, human scores.
