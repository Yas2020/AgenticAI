"""Format offline evaluation reports."""

import json
from pathlib import Path


def format_report(metrics: dict) -> str:
    success_pct = metrics["success_rate"] * 100
    tool_usage_pct = metrics["tool_usage_rate"] * 100
    dag_failure_pct = metrics["dag_failure_rate"] * 100
    budget_failure_pct = metrics["budget_failure_rate"] * 100
    failures = metrics.get("top_failure_modes") or ["None"]

    lines = [
        "================================",
        "OFFLINE EVAL REPORT",
        "================================",
        f"Success Rate: {success_pct:.0f}%",
        f"Average Faithfulness: {metrics['average_faithfulness']:.2f}",
        f"Average Completeness: {metrics['average_completeness']:.2f}",
        f"Budget Failure Rate: {budget_failure_pct:.0f}%",
        f"Tool Usage Rate: {tool_usage_pct:.0f}%",
        f"DAG Failure Rate: {dag_failure_pct:.0f}%",
        f"Average Latency: {metrics['average_latency_ms']:.0f} ms",
        (
            "Token Usage: "
            f"{metrics['token_usage']['input_tokens']} in / "
            f"{metrics['token_usage']['output_tokens']} out"
        ),
        f"Estimated Cost: ${metrics['estimated_cost_usd']:.4f}",
        "",
        "Top Failure Modes:",
    ]
    lines.extend(f"- {item}" for item in failures)
    lines.append("================================")
    return "\n".join(lines)


def write_report(metrics: dict, results: list, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "latest.json"
    payload = {"metrics": metrics, "results": results}
    report_path.write_text(json.dumps(payload, indent=2, default=str))
    text_path = output_dir / "latest.txt"
    text_path.write_text(format_report(metrics))
    return report_path
