"""Rate-limit helpers for integration evals."""

import asyncio
import os

DEFAULT_CASE_DELAY = float(os.getenv("EVAL_CASE_DELAY_SECONDS", "5"))
DEFAULT_MAX_RETRIES = int(os.getenv("EVAL_RATE_LIMIT_RETRIES", "3"))


def is_rate_limit_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "rate limit" in message or "429" in message or "too many requests" in message


async def sleep_between_cases(delay: float | None = None) -> None:
    await asyncio.sleep(delay if delay is not None else DEFAULT_CASE_DELAY)


async def run_with_retry(coro_factory, max_retries: int | None = None):
    """Retry async call on OpenAI rate-limit errors with exponential backoff."""
    retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
    delay = DEFAULT_CASE_DELAY

    for attempt in range(retries):
        try:
            return await coro_factory()
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt == retries - 1:
                raise
            wait = delay * (2**attempt)
            await asyncio.sleep(wait)
