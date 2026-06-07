"""Approximate OpenAI pricing used for eval reports and Langfuse cost ingestion."""

from typing import Any

MODEL_PRICING_PER_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "default": {"input": 1.00, "output": 3.00},
}


def _pricing_for_model(model: str | None) -> dict[str, float]:
    name = (model or "").lower()
    if "gpt-4.1-nano" in name or ("gpt-4.1" in name and "nano" in name):
        return MODEL_PRICING_PER_1M["gpt-4.1-nano"]
    if "gpt-4o" in name:
        return MODEL_PRICING_PER_1M["gpt-4o"]
    return MODEL_PRICING_PER_1M["default"]


def _normalize_token_counts(usage: dict[str, Any] | None) -> tuple[int, int]:
    if not usage:
        return 0, 0

    inp = (
        usage.get("input")
        or usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or 0
    )
    out = (
        usage.get("output")
        or usage.get("completion_tokens")
        or usage.get("output_tokens")
        or 0
    )
    return int(inp), int(out)


def estimate_cost_details(
    usage: dict[str, Any] | None,
    model: str | None = None,
) -> dict[str, float] | None:
    """Return Langfuse cost_details with a canonical ``total`` key."""
    input_tokens, output_tokens = _normalize_token_counts(usage)
    if input_tokens == 0 and output_tokens == 0:
        return None

    pricing = _pricing_for_model(model)
    input_cost = round((input_tokens / 1_000_000) * pricing["input"], 8)
    output_cost = round((output_tokens / 1_000_000) * pricing["output"], 8)
    return {
        "input": input_cost,
        "output": output_cost,
        "total": round(input_cost + output_cost, 8),
    }


def estimate_cost_usd(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str | None = None,
) -> float:
    details = estimate_cost_details(
        {"input": input_tokens, "output": output_tokens},
        model=model,
    )
    return float(details["total"]) if details else 0.0
