"""PC 側: ベンチマーク問題の読み込み、プロンプト生成、採点.

プロンプトと採点は scripts/test_*.py と同一。scripts 側は energy_profiler (pyRAPL) と
ollama_utils (Python 3.10+ 構文) に依存して PC の Python 3.9 環境では import できないため、
必要な部分をここに持つ。

Python 3.9 互換.
"""
from __future__ import annotations

import gzip
import json
import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("bench_tasks")

DATASETS = ["human_eval", "gsm8k", "mmlu-pro"]


@dataclass
class Task:
    dataset: str
    task_id: str
    prompt: str
    meta: dict = field(default_factory=dict)  # レコードに含める問題情報
    gold: Optional[str] = None


def _read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_id_file(path: Optional[str]) -> Optional[set]:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


# ---------------------------------------------------------------- HumanEval

HUMAN_EVAL_INSTRUCTION = (
    "Please complete the following Python code without providing any additional tasks such as testing or explanations\n"
)


def _load_human_eval(path: Optional[str]) -> List[Task]:
    if path is not None:
        rows = list(_read_jsonl(Path(path)))
    else:
        from datasets import load_dataset

        rows = list(load_dataset("openai/openai_humaneval", split="test"))
    return [Task("human_eval", r["task_id"], HUMAN_EVAL_INSTRUCTION + r["prompt"]) for r in rows]


# ---------------------------------------------------------------- GSM8K

ANS_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _to_decimal(value: str):
    cleaned = value.replace(",", "").strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def gsm8k_extract_answer(text: str):
    m = ANS_RE.search(text)
    if m:
        return m.group(1)
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def _gsm8k_is_correct(pred_text: str, gold_text: str) -> bool:
    pred = gsm8k_extract_answer(pred_text)
    gold = gsm8k_extract_answer(gold_text)
    if pred is None or gold is None:
        return False
    pred_dec = _to_decimal(pred)
    gold_dec = _to_decimal(gold)
    if pred_dec is None or gold_dec is None:
        return False
    return pred_dec == gold_dec


def _gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following grade school math problem.\n"
        "Return the final answer in the format: #### <number>\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _load_gsm8k(path: Optional[str], selected_idx_file: Optional[str]) -> List[Task]:
    if path is not None:
        rows = list(_read_jsonl(Path(path)))
    else:
        from datasets import load_dataset

        # grade-school-math の test.jsonl と同じ並び
        rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    selected = _read_id_file(selected_idx_file)
    tasks = []
    for i, r in enumerate(rows):
        if selected is not None and i not in selected:
            continue
        tasks.append(
            Task(
                "gsm8k",
                f"gsm8k/{i}",
                _gsm8k_prompt(r["question"]),
                meta={"question": r["question"], "gold_answer": gsm8k_extract_answer(r["answer"])},
                gold=r["answer"],
            )
        )
    return tasks


# ---------------------------------------------------------------- MMLU-Pro

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

# MMLU-Pro リポジトリの cot_prompt_lib/initial_prompt.txt が見つからない場合に使う
FALLBACK_INITIAL_PROMPT = (
    "The following are multiple choice questions (with answers) about {$}. Think step by step and then "
    'finish your answer with "the answer is (X)" where X is the correct letter choice.\n'
)


def _mmlu_format_question(example: dict) -> str:
    prompt = "Question:\n" + example["question"] + "\n" + "Options:\n"
    for i, opt in enumerate(example["options"]):
        prompt += "{}. {}\n".format(CHOICES[i], opt)
    prompt += "Answer: Let's think step by step."
    return prompt


def mmlu_extract_answer(response: str):
    if response is None:
        return None
    patterns = [
        r"The answer is\s*\(?([A-J])\)?",
        r"the answer is\s*\(?([A-J])\)?",
        r"answer is\s*\(?([A-J])\)?",
        r"Answer\s*:\s*\(?([A-J])\)?",
        r"Final answer\s*:\s*\(?([A-J])\)?",
        r"\(([A-J])\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    matches = re.findall(r"\b([A-J])\b", response, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None


def _load_mmlu_pro(mmlu_pro_dir: Optional[str], selected_ids_file: Optional[str]) -> List[Task]:
    from datasets import load_dataset

    initial = None
    if mmlu_pro_dir is not None:
        p = Path(mmlu_pro_dir) / "cot_prompt_lib" / "initial_prompt.txt"
        if p.exists():
            initial = p.read_text(encoding="utf-8")
    if initial is None:
        log.warning("MMLU-Pro initial_prompt.txt not found; using built-in copy.")
        initial = FALLBACK_INITIAL_PROMPT

    selected = _read_id_file(selected_ids_file)
    tasks = []
    # scripts/test_mmlu_pro.py と同じく ntrain=0 (few-shot 例なし)
    for e in load_dataset("TIGER-Lab/MMLU-Pro", split="test"):
        if selected is not None and e["question_id"] not in selected:
            continue
        e = dict(e)
        e["options"] = [opt for opt in e["options"] if opt != "N/A"]
        prompt = initial.replace("{$}", e["category"]) + "\n" + _mmlu_format_question(e)
        tasks.append(
            Task(
                "mmlu-pro",
                f"mmlu-pro/{e['question_id']}",
                prompt,
                meta={
                    "question_id": e["question_id"],
                    "category": e["category"],
                    "question": e["question"],
                    "options": e["options"],
                    "answer": e["answer"],
                    "answer_index": e["answer_index"],
                    "ntrain": 0,
                },
                gold=e["answer"],
            )
        )
    return tasks


# ---------------------------------------------------------------- 共通


def load_tasks(
    dataset: str,
    max_samples: Optional[int] = None,
    human_eval_path: Optional[str] = None,
    gsm8k_path: Optional[str] = None,
    gsm8k_selected_idx_file: Optional[str] = None,
    mmlu_pro_dir: Optional[str] = None,
    mmlu_pro_selected_ids_file: Optional[str] = None,
) -> List[Task]:
    if dataset == "human_eval":
        tasks = _load_human_eval(human_eval_path)
    elif dataset == "gsm8k":
        tasks = _load_gsm8k(gsm8k_path, gsm8k_selected_idx_file)
    elif dataset == "mmlu-pro":
        tasks = _load_mmlu_pro(mmlu_pro_dir, mmlu_pro_selected_ids_file)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if max_samples is not None:
        tasks = tasks[:max_samples]
    return tasks


def score(task: Task, completion: str) -> dict:
    """採点結果 (HumanEval は human-eval の evaluate_functional_correctness で後から評価する)."""
    if task.dataset == "gsm8k":
        return {"pred_answer": gsm8k_extract_answer(completion), "correct": _gsm8k_is_correct(completion, task.gold)}
    if task.dataset == "mmlu-pro":
        pred = mmlu_extract_answer(completion)
        return {"pred": pred, "correct": pred == task.gold}
    return {}
