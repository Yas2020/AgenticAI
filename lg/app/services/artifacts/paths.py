"""Shared artifact directory helpers for prod, docker, and local pytest."""

import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def artifact_root() -> Path:
    """Root directory for all run artifacts."""
    env_path = os.getenv("ARTIFACT_DIR")
    if env_path:
        return Path(env_path)
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "shared-artifacts"


def _slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:max_len] or "report"


def final_report_paths(topic: str, query: str | None = None) -> tuple[Path, Path]:
    root = artifact_root() / "final_reports"
    root.mkdir(parents=True, exist_ok=True)

    slug = _slugify(query or topic or "report")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    report_dir = root / f"{timestamp}_{slug}"
    report_dir.mkdir(parents=True, exist_ok=True)

    archived = report_dir / "final_report.md"
    latest = root / "latest.md"
    return archived, latest


def _artifact_fields(artifact: Any) -> tuple[str | None, dict | None, int | None]:
    if isinstance(artifact, dict):
        return (
            artifact.get("artifact_type"),
            artifact.get("content") if isinstance(artifact.get("content"), dict) else None,
            artifact.get("task_id"),
        )
    return (
        getattr(artifact, "artifact_type", None),
        artifact.content if isinstance(getattr(artifact, "content", None), dict) else None,
        getattr(artifact, "task_id", None),
    )


def attach_quant_outputs(artifacts: list[Any], report_dir: Path) -> str:
    """
    Copy quant plots/scripts into report_dir/quant/ and return a markdown appendix.
    """
    quant_dir = report_dir / "quant"
    lines: list[str] = []
    root = artifact_root()

    for artifact in artifacts or []:
        artifact_type, content, task_id = _artifact_fields(artifact)
        if artifact_type != "quantitative_analyst" or not content:
            continue

        prefix = f"task_{task_id}" if task_id is not None else "quant"

        run_id = content.get("run_id")
        if run_id:
            script_src = root / f"run_{run_id}" / "script.py"
            if script_src.exists():
                quant_dir.mkdir(parents=True, exist_ok=True)
                script_dest = quant_dir / f"{prefix}_script.py"
                shutil.copy2(script_src, script_dest)
                lines.append(f"- Analysis script: [`{script_dest.name}`](quant/{script_dest.name})")

        for plot in content.get("plots") or []:
            rel_path = plot.get("rel_path") or plot.get("name")
            if not rel_path:
                continue
            plot_src = root / rel_path
            if not plot_src.exists():
                continue
            quant_dir.mkdir(parents=True, exist_ok=True)
            plot_name = plot.get("name") or plot_src.name
            plot_dest = quant_dir / f"{prefix}_{plot_name}"
            shutil.copy2(plot_src, plot_dest)
            lines.append(f"![{plot_name}](quant/{plot_dest.name})")

    if not lines:
        return ""

    return "## Quantitative Visualizations\n\n" + "\n\n".join(lines)


def write_final_report(
    content: str,
    *,
    topic: str,
    query: str | None = None,
    model_name: str,
    artifacts: list[Any] | None = None,
) -> Path:
    """Write report to archived folder and refresh latest.md. Returns archived path."""
    archived, latest = final_report_paths(topic=topic, query=query)
    report_dir = archived.parent

    quant_section = attach_quant_outputs(artifacts or [], report_dir)

    header = f"""---
topic: {topic}
model: {model_name}
timestamp: {datetime.now(timezone.utc).isoformat()}
archived_path: {archived.relative_to(artifact_root())}
latest_path: {latest.relative_to(artifact_root())}
---

"""
    body = header + content
    if quant_section:
        body = body + "\n\n" + quant_section

    archived.write_text(body)

    latest_body = body
    if quant_section:
        latest_body = header + content + "\n\n" + quant_section.replace(
            "](quant/", f"]({report_dir.name}/quant/"
        )
    latest.write_text(latest_body)
    return archived
