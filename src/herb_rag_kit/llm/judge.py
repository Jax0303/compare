from __future__ import annotations
from typing import Tuple
from .gemini_client import generate

JUDGE_PROMPT = """
You are a judge. Given QUESTION, ANSWER, and EVIDENCE, return two floats:
Correctness in [-1,2] and Faithfulness in [-1,1].
Respond as: correctness=<value> faithfulness=<value>
QUESTION: {q}
ANSWER: {a}
EVIDENCE: {ev}
"""

def judge_correctness_faithfulness(question: str, answer: str, evidence: str) -> Tuple[float, float]:
    txt = generate(JUDGE_PROMPT.format(q=question, a=answer, ev=evidence), temperature=0.0, max_output_tokens=64)
    import re
    c = re.search(r"correctness\s*=\s*([-+]?[0-9]*\.?[0-9]+)", txt)
    f = re.search(r"faithfulness\s*=\s*([-+]?[0-9]*\.?[0-9]+)", txt)
    cval = float(c.group(1)) if c else 0.0
    fval = float(f.group(1)) if f else 0.0
    return cval, fval
