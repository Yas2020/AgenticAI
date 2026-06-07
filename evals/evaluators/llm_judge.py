"""Single-pass LLM judge — four compliance pillars in one call."""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.services.artifacts.evidence import format_evidence_bundle
from evals.evaluators.defaults import resolve_llm_checks
from evals.test_cases.schemas import EvalCase, EvalRunResult, EvaluatorResult, LLMMetric

JUDGE_PROMPT = """You are an elite financial auditor performing a single-pass compliance review on an AI investment analyst's report.

Evaluate the report across the four pillars defined in the structured schema:
1. FAITHFULNESS (0.0-1.0): Cross-reference every number, stat, and percentage with the Ground Truth evidence. Penalize unsourced assertions. Reward explicit uncertainty when evidence is sparse. Unit equivalents (39.1B vs 39,100M) are acceptable.
2. CITATION FIDELITY (0.0-1.0): If the report uses citations, markdown links, or source tags, verify they align with the evidence. Score 1.0 when no citations are present and claims are still faithful.
3. INTERNAL CONSISTENCY (0.0-1.0): Flag conflicting numbers, units, or narratives within the report itself.
4. COMPLETENESS (0.0-1.0): Judge whether the report adequately answers the user's query with appropriate depth — not a fixed template.

Analyze the evidence and final report carefully. Output the structured JSON scoring model only.
"""


class JudgeScores(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    citation_fidelity: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    rationale: str = ""


def _build_judge_input(case: EvalCase, run: EvalRunResult) -> str:
    max_chars = int(os.getenv("EVAL_EVIDENCE_MAX_CHARS", "12000"))
    evidence = format_evidence_bundle(
        run.artifacts,
        exclude_types={"final_report"},
        max_chars=max_chars,
    )
    return f"""Case ID: {case.id}
Category: {case.category}
User query: {case.input_query}
Execution path: {run.execution_path}

GROUND TRUTH EVIDENCE (upstream artifacts only — no final report):
{evidence or "(no upstream evidence)"}

FINAL REPORT:
{run.final_response}
"""


def evaluate_llm_judge(case: EvalCase, run: EvalRunResult | None) -> EvaluatorResult | None:
    llm_checks = resolve_llm_checks(case)
    if not llm_checks:
        return None

    if not run or not run.final_response:
        if case.run_full_graph:
            return EvaluatorResult(
                name="llm_judge",
                passed=False,
                score=0.0,
                explanation="Graph run not completed. No final report to judge.",
            )
        return EvaluatorResult(
            name="llm_judge",
            passed=True,
            score=1.0,
            explanation="Skipped — no full graph run for this case.",
        )

    if not os.getenv("OPENAI_API_KEY"):
        return EvaluatorResult(
            name="llm_judge",
            passed=True,
            score=1.0,
            explanation="Skipped — OPENAI_API_KEY not set.",
        )

    model = ChatOpenAI(
        model=os.getenv("EVAL_JUDGE_MODEL", "gpt-4.1-nano"),
        temperature=0,
    )
    structured = model.with_structured_output(JudgeScores)
    scores = structured.invoke(
        [
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=_build_judge_input(case, run)),
        ]
    )

    score_map: dict[LLMMetric, float] = {
        "faithfulness": scores.faithfulness,
        "citation_fidelity": scores.citation_fidelity,
        "consistency": scores.consistency,
        "completeness": scores.completeness,
    }

    metric_results: dict[str, dict] = {}
    failures: list[str] = []
    for check in llm_checks:
        actual = score_map[check.metric]
        passed = actual >= check.min_score
        metric_results[check.metric] = {
            "score": round(actual, 3),
            "min_score": check.min_score,
            "passed": passed,
        }
        if not passed:
            failures.append(f"{check.metric}={actual:.2f} < {check.min_score}")

    passed = not failures
    avg_score = sum(score_map[m] for m in score_map) / len(score_map)
    return EvaluatorResult(
        name="llm_judge",
        passed=passed,
        score=round(avg_score, 3),
        explanation=scores.rationale if passed else "; ".join(failures),
        details={
            "metrics": metric_results,
            "unsupported_claims": scores.unsupported_claims,
            "rationale": scores.rationale,
        },
    )
