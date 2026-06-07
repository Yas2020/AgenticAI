"""Pytest configuration and fixtures for offline evaluation."""

import os
import sys
from pathlib import Path

import asyncio
import pytest
import pytest_asyncio
from langgraph.checkpoint.memory import MemorySaver

ROOT = Path(__file__).resolve().parents[1]
LG_ROOT = ROOT / "lg"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LG_ROOT))

from app.core.engine import build_graph  # noqa: E402
from app.services.mcp.mcp_clients import mcp_manager  # noqa: E402
from app.services.observability.langfuse_tracer import (  # noqa: E402
    ensure_langfuse_initialized,
    flush_langfuse,
)
from evals.metrics.aggregator import aggregate_metrics  # noqa: E402
from evals.metrics.report import write_report  # noqa: E402
from evals.test_cases.schemas import CaseEvalResult, EvalCase  # noqa: E402

DATASET_PATH = Path(__file__).parent / "datasets" / "benchmark.jsonl"
REPORT_DIR = Path(__file__).parent / "reports"
MCP_STARTUP_TIMEOUT = float(os.getenv("MCP_STARTUP_TIMEOUT", "30"))


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration evals that require live services and API keys.",
    )
    parser.addoption(
        "--enforce-thresholds",
        action="store_true",
        default=False,
        help="Fail the run when aggregate metrics fall below thresholds.",
    )
    parser.addoption(
        "--max-integration-cases",
        type=int,
        default=int(os.getenv("EVAL_MAX_INTEGRATION_CASES", "3")),
        help="Max integration cases to run (0 = all). Default 3 to reduce rate limits.",
    )
    parser.addoption(
        "--run-full-suite",
        action="store_true",
        default=False,
        help="Run all integration cases (overrides --max-integration-cases).",
    )
    parser.addoption(
        "--eval-case-delay",
        type=float,
        default=float(os.getenv("EVAL_CASE_DELAY_SECONDS", "5")),
        help="Seconds to wait between integration cases.",
    )
    parser.addoption(
        "--eval-case-id",
        action="append",
        default=[],
        help="Run only these benchmark case IDs (repeatable). Example: --eval-case-id simple-001",
    )


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "unit: fast offline eval checks")
    config.addinivalue_line("markers", "integration: full graph eval pipeline")


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(
        reason="Integration evals require --run-integration."
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def load_benchmark_cases() -> list[EvalCase]:
    cases = []
    for line in DATASET_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            cases.append(EvalCase.model_validate_json(line))
    return cases


def filter_integration_cases(cases: list[EvalCase], config) -> list[EvalCase]:
    integration = [case for case in cases if case.run_full_graph]

    case_ids = config.getoption("--eval-case-id") or []
    if case_ids:
        allowed = set(case_ids)
        integration = [case for case in integration if case.id in allowed]

    if config.getoption("--run-full-suite"):
        return integration
    max_cases = config.getoption("--max-integration-cases")
    if max_cases <= 0:
        return integration
    return integration[:max_cases]


@pytest.fixture(scope="session")
def benchmark_cases() -> list[EvalCase]:
    return load_benchmark_cases()


@pytest.fixture(scope="session")
def code_cases(benchmark_cases) -> list[EvalCase]:
    return [case for case in benchmark_cases if not case.run_full_graph]


@pytest.fixture(scope="session")
def integration_cases(request, benchmark_cases) -> list[EvalCase]:
    return filter_integration_cases(benchmark_cases, request.config)


@pytest.fixture(scope="session")
def eval_report_dir() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR


@pytest.fixture(scope="session", autouse=True)
def eval_env():
    os.environ.setdefault("EVAL_MODE", "1")
    os.environ.setdefault("ARTIFACT_DIR", str(ROOT / "shared-artifacts"))
    os.environ.setdefault("LANGFUSE_OTEL_INGESTION_VERSION", "4")
    ensure_langfuse_initialized()
    yield
    flush_langfuse()


@pytest_asyncio.fixture
async def eval_graph():
    graph = build_graph(checkpointer=MemorySaver())
    yield graph


@pytest_asyncio.fixture
async def mcp_lifecycle(request):
    if not request.config.getoption("--run-integration"):
        yield
        return

    started: list[str] = []
    failures: list[str] = []

    for name, manager in mcp_manager.items():
        try:
            await manager.startup(timeout=MCP_STARTUP_TIMEOUT)
            started.append(name)
        except Exception as exc:
            failures.append(f"{name} ({manager.url}): {exc}")

    if failures:
        for name in started:
            await mcp_manager[name].shutdown()
        pytest.skip(
            "MCP services unavailable. Start with "
            "`docker compose up -d mcp-research mcp-quant`. "
            + "; ".join(failures)
        )

    yield

    for name in reversed(started):
        await mcp_manager[name].shutdown()
        await asyncio.sleep(0)


@pytest.fixture
def evaluate_case():
    from evals.evaluators.evaluate import evaluate_case as run_evaluators

    return run_evaluators


@pytest.fixture
def finalize_report(eval_report_dir):
    def _finalize(results: list[CaseEvalResult], request) -> dict:
        serializable = [
            {
                "case_id": result.case.id,
                "category": result.case.category,
                "evaluators": [item.model_dump() for item in result.evaluators],
                "run": result.run.model_dump() if result.run else None,
                "failure_modes": result.failure_modes,
            }
            for result in results
        ]
        metrics = aggregate_metrics(results)
        write_report(metrics, serializable, eval_report_dir)
        print("\n" + (eval_report_dir / "latest.txt").read_text())

        if request.config.getoption("--enforce-thresholds"):
            if metrics["success_rate"] < 0.70:
                pytest.fail(f"Success rate below threshold: {metrics['success_rate']}")
        return metrics

    return _finalize
