"""Deterministic NMR-Bench metrics, independent of inference libraries."""

from .counting import check_correct, extract_gt_from_answer, parse_answer
from .path import extract_boxed_pairs, normalize_pairs, strip_thinking

TASKS = ("burden_counting", "path_tracing")
COUNT_FORMS = {"T1a", "T2a", "T2b", "T2c", "T2d"}


def score_text(task, sample, text, reasoning=""):
    if not isinstance(text, str) or not isinstance(reasoning, str):
        raise TypeError("Predictions and reasoning must be strings")
    if task == "path_tracing":
        pred, status = extract_boxed_pairs(strip_thinking(text))
        target, target_status = extract_boxed_pairs(sample["answer"])
        if target_status != "success":
            raise ValueError(f"Invalid benchmark answer: {sample['sample_id']}")
        truth = normalize_pairs(target)
        valid = status == "success"
        guessed = normalize_pairs(pred) if valid else frozenset()
        labels = set().union(*truth)
        if valid and not set().union(*guessed).issubset(labels):
            valid, status = False, "label_unknown"
        return {
            "correct": valid and guessed == truth,
            "invalid": not valid,
            "cra": len(guessed & truth) / len(truth) if valid else 0.0,
            "parse_status": status,
        }
    if task != "burden_counting":
        raise ValueError(f"Unknown task: {task}")
    content = text
    if not reasoning and "<｜end▁of▁thinking｜>" in text:
        parts = text.split(" response", 1)
        reasoning, content = parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
    form = sample["task_form"]
    parsed = parse_answer(content, reasoning, form)
    target, _ = extract_gt_from_answer(sample["answer"], form)
    if target is None:
        raise ValueError(f"Invalid benchmark answer: {sample['sample_id']}")
    correct, error = check_correct(parsed["parsed_value"], target, form)
    return {
        "correct": bool(correct),
        "invalid": bool(parsed["parse_fail"]),
        "absolute_error": error,
        "pure_count": form in COUNT_FORMS,
        "parse_status": parsed["parse_fail_reason"] or "success",
    }


def summarize(rows, task):
    def group(items):
        n = len(items)
        result = {
            "n": n,
            "accuracy": 100 * sum(r["correct"] for r in items) / n,
            "invalid": sum(r["invalid"] for r in items),
        }
        if task == "path_tracing":
            result["cra"] = 100 * sum(r["cra"] for r in items) / n
        else:
            numerical = [
                r["absolute_error"]
                for r in items
                if r["pure_count"] and not r["invalid"] and r["absolute_error"] >= 0
            ]
            result["mae_n"] = len(numerical)
            result["mae"] = sum(numerical) / len(numerical) if numerical else None
        return result

    if not rows:
        raise ValueError("No predictions to score")
    return {
        "task": task,
        "percentage_metrics": ["accuracy", "cra"] if task == "path_tracing" else ["accuracy"],
        "overall": group(rows),
        "by_difficulty": {
            level: group([r for r in rows if r["difficulty"] == level])
            for level in ("easy", "medium", "hard")
            if any(r["difficulty"] == level for r in rows)
        },
    }
