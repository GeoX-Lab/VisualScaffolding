"""Answer extraction for the seven Burden Counting task forms."""

from __future__ import annotations

import re


def _normalize_label(label: str) -> str:
    return " ".join(label.lower().strip().split())


def _extract_boxed(text: str) -> list[str]:
    return re.findall("\\\\boxed\\{([^}]+)\\}", text)


def _parse_boxed_value(raw: str, task_form: str) -> dict:
    if task_form in ("T1a", "T2a", "T2b", "T2c", "T2d"):
        try:
            return {"parsed_value": int(raw), "parse_fail": False, "parse_fail_reason": ""}
        except ValueError:
            nums = re.findall("\\d+", raw)
            if nums:
                return {"parsed_value": int(nums[-1]), "parse_fail": False, "parse_fail_reason": ""}
            return {"parsed_value": None, "parse_fail": True, "parse_fail_reason": f"not_int:{raw}"}
    if task_form == "T3a":
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 2:
            size = parts[0].strip().lower()
            try:
                diff = int(parts[1].strip())
                return {"parsed_value": (size, diff), "parse_fail": False, "parse_fail_reason": ""}
            except ValueError:
                return {
                    "parsed_value": None,
                    "parse_fail": True,
                    "parse_fail_reason": f"t3a_diff_not_int:{raw}",
                }
        return {"parsed_value": None, "parse_fail": True, "parse_fail_reason": f"t3a_format:{raw}"}
    if task_form == "T3b":
        parts = [p.strip() for p in raw.rsplit(",", 1)]
        if len(parts) == 2:
            label = _normalize_label(parts[0])
            try:
                count = int(parts[1])
                return {
                    "parsed_value": (label, count),
                    "parse_fail": False,
                    "parse_fail_reason": "",
                }
            except ValueError:
                pass
        parts_space = raw.rsplit(None, 1)
        if len(parts_space) == 2:
            try:
                count = int(parts_space[1])
                label = _normalize_label(parts_space[0])
                return {
                    "parsed_value": (label, count),
                    "parse_fail": False,
                    "parse_fail_reason": "",
                }
            except ValueError:
                try:
                    count = int(parts_space[0])
                    label = _normalize_label(parts_space[1])
                    return {
                        "parsed_value": (label, count),
                        "parse_fail": False,
                        "parse_fail_reason": "",
                    }
                except ValueError:
                    pass
        return {"parsed_value": None, "parse_fail": True, "parse_fail_reason": f"t3b_format:{raw}"}
    return {
        "parsed_value": None,
        "parse_fail": True,
        "parse_fail_reason": f"unknown_form:{task_form}",
    }


def extract_gt_from_answer(answer: str, task_form: str):
    if not answer:
        return (None, None)
    boxes = _extract_boxed(answer)
    if not boxes:
        return (None, None)
    raw = boxes[-1].strip()
    result = _parse_boxed_value(raw, task_form)
    return (result.get("parsed_value"), raw)


def _reasoning_fallback(reasoning: str, task_form: str) -> dict:
    if not reasoning:
        return {
            "parsed_value": None,
            "parse_fail": True,
            "parse_fail_reason": "no_content_no_reasoning",
            "parsed_from": "none",
        }
    boxes = _extract_boxed(reasoning)
    if boxes:
        result = _parse_boxed_value(boxes[-1].strip(), task_form)
        result["parsed_from"] = "reasoning_boxed"
        return result
    lines = reasoning.strip().split("\n")
    tail_lines = lines[-30:]
    tail_text = "\n".join(tail_lines)
    patterns = [
        (
            "(?:total|overall|altogether|in\\s+total|final\\s+count|final\\s+answer|sum\\s+(?:is|=))\\D*(\\d+)",
            "pattern_total",
        ),
        (
            "(?:there\\s+(?:are|is)|that\\'?s?\\s+)\\s*(\\d+)\\s+(?:in\\s+total|altogether|overall|total)",
            "pattern_there_are",
        ),
        ("(?:I\\s+count|I\\s+counted|we\\s+have|we\\s+get)\\s+(\\d+)", "pattern_i_count"),
        (
            "(?:answer\\s+(?:is|=|:)|so\\s+(?:the\\s+)?answer\\s+(?:is|=|:))\\s*(\\d+)",
            "pattern_answer_is",
        ),
    ]
    for pat, pat_name in patterns:
        matches = re.findall(pat, tail_text, re.IGNORECASE)
        if matches:
            val = int(matches[-1])
            return {
                "parsed_value": val,
                "parse_fail": False,
                "parse_fail_reason": "",
                "parsed_from": pat_name,
            }
    for line in reversed(tail_lines[-5:]):
        nums = re.findall("\\b(\\d+)\\b", line)
        if nums:
            val = int(nums[-1])
            return {
                "parsed_value": val,
                "parse_fail": False,
                "parse_fail_reason": "",
                "parsed_from": "last_integer",
            }
    return {
        "parsed_value": None,
        "parse_fail": True,
        "parse_fail_reason": "no_boxed_no_pattern",
        "parsed_from": "none",
    }


def parse_answer(content: str, reasoning: str, task_form: str) -> dict:
    if content:
        boxes = _extract_boxed(content)
        if boxes:
            result = _parse_boxed_value(boxes[-1].strip(), task_form)
            result["parsed_from"] = "content_boxed"
            result["boxed_content"] = boxes
            return result
    fb = _reasoning_fallback(reasoning, task_form)
    fb["boxed_content"] = _extract_boxed(reasoning) if reasoning else []
    return fb


def check_correct(parsed, gt_value, task_form: str) -> tuple[bool, int]:
    if parsed is None or gt_value is None:
        return (False, -1)
    if task_form in ("T1a", "T2a", "T2b", "T2c", "T2d"):
        try:
            ae = abs(parsed - int(gt_value))
            return (parsed == int(gt_value), ae)
        except (TypeError, ValueError):
            return (False, -1)
    if task_form == "T3a":
        if (
            isinstance(parsed, tuple)
            and isinstance(gt_value, tuple)
            and (len(parsed) == 2)
            and (len(gt_value) == 2)
        ):
            p_winner, p_diff = parsed
            g_winner, g_diff = gt_value
            correct = p_winner == g_winner and p_diff == g_diff
            ae = abs(p_diff - g_diff)
            return (correct, ae)
        return (False, -1)
    if task_form == "T3b":
        if (
            isinstance(parsed, tuple)
            and isinstance(gt_value, tuple)
            and (len(parsed) == 2)
            and (len(gt_value) == 2)
        ):
            p_label, p_count = parsed
            g_label, g_count = gt_value
            correct = p_label == g_label and p_count == g_count
            ae = abs(p_count - g_count)
            return (correct, ae)
        return (False, -1)
    return (False, -1)
