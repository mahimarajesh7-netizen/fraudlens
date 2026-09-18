"""Turn a SHAP local explanation into a plain-language write-up via the Gemini API.

Design decision: this layer is deliberately NOT asked to reason about *why* a transaction looks
fraudulent. SHAP already computed that, exactly, from the trained model. Gemini's only job is to
translate the SHAP output (feature, value, direction, magnitude) into a sentence a fraud analyst
or customer-support agent can read in two seconds — grounded strictly in the numbers it's given,
with a system instruction against inventing reasons that aren't in the data. In a fraud/compliance
context, an LLM that free-associates plausible-sounding-but-fabricated reasons for a flag is a
liability, not a feature.
"""
from __future__ import annotations

import os
import requests

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

SYSTEM_INSTRUCTION = (
    "You are FraudLens' explanation layer. You are given a fraud model's output probability and "
    "its top SHAP feature attributions for one transaction. Write a 2-4 sentence plain-language "
    "explanation for a fraud analyst who is not a data scientist. Rules: only reference the "
    "features and directions given to you — never invent a reason not present in the data. State "
    "the risk level and probability plainly, then explain the top 2-3 drivers in business terms. "
    "Do not use technical jargon like 'SHAP value' or 'feature vector' in the output."
)


def _format_drivers(top_drivers: list[dict]) -> str:
    lines = []
    for d in top_drivers:
        direction = "pushes toward FRAUD" if d["shap"] > 0 else "pushes toward LEGITIMATE"
        lines.append(f"- {d['feature']} = {d['value']}  ({direction}, strength {abs(d['shap']):.3f})")
    return "\n".join(lines)


def explain_in_plain_language(explanation: dict, api_key: str | None = None) -> str:
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    prompt = (
        f"Fraud probability: {explanation['fraud_probability']:.1%}\n"
        f"Top feature drivers (sorted by influence):\n"
        f"{_format_drivers(explanation['top_drivers'])}\n\n"
        "Write the plain-language explanation now."
    )

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        # thinkingBudget=0: this is a constrained rewrite of numbers we already computed, not a
        # reasoning task — thinking tokens would just eat the output budget for no benefit.
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 400,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    resp = requests.post(
        GEMINI_URL,
        params={"key": api_key},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
