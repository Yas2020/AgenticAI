"""Format upstream agent artifacts into evidence bundles for synthesis and evaluation."""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

DEFAULT_EXCLUDE_TYPES = frozenset({"final_report"})


def _get(artifact: Any, key: str, default: Any = None) -> Any:
    if isinstance(artifact, dict):
        return artifact.get(key, default)
    return getattr(artifact, key, default)


def filter_evidence_artifacts(
    artifacts: Sequence[Any] | None,
    *,
    sources: Sequence[str] | None = None,
    exclude_types: Iterable[str] | None = None,
) -> list[Any]:
    artifacts = artifacts or []
    excluded = set(exclude_types if exclude_types is not None else DEFAULT_EXCLUDE_TYPES)
    filtered: list[Any] = []
    for artifact in artifacts:
        artifact_type = _get(artifact, "artifact_type")
        if artifact_type in excluded:
            continue
        source = _get(artifact, "source")
        if sources is not None and source not in sources:
            continue
        filtered.append(artifact)
    return filtered


def _format_content(content: Any) -> str:
    if isinstance(content, dict):
        return "\n".join(f"- {k}: {v}" for k, v in content.items())
    if content is None:
        return ""
    return str(content)


def format_evidence_bundle(
    artifacts: Sequence[Any] | None,
    *,
    sources: Sequence[str] | None = None,
    exclude_types: Iterable[str] | None = None,
    max_chars: int | None = None,
) -> str:
    blocks: list[str] = []
    for artifact in filter_evidence_artifacts(
        artifacts, sources=sources, exclude_types=exclude_types
    ):
        source = _get(artifact, "source", "unknown")
        artifact_type = _get(artifact, "artifact_type", "artifact")
        content = _format_content(_get(artifact, "content"))
        if not content.strip():
            continue
        blocks.append(f"[{source}/{artifact_type}]\n{content}")

    bundle = "\n\n".join(blocks)
    if max_chars and len(bundle) > max_chars:
        bundle = bundle[: max_chars - 20] + "\n...[truncated]"
    return bundle


def build_source_citations(artifacts: Sequence[Any] | None) -> tuple[str, list[dict[str, Any]]]:
    """Numbered source list for analyst [^n] citations from research artifacts."""
    entries: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for artifact in filter_evidence_artifacts(artifacts, sources=("research",)):
        content = _get(artifact, "content")
        if not isinstance(content, dict):
            continue
        task_id = _get(artifact, "task_id")
        for url in content.get("sources") or []:
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            entries.append(
                {
                    "id": len(entries) + 1,
                    "url": str(url),
                    "task_id": task_id,
                }
            )

    if not entries:
        return "(no external sources captured in research artifacts)", entries

    lines = [f"[{entry['id']}] {entry['url']}" for entry in entries]
    return "\n".join(lines), entries


def collect_grounding_text(artifacts: Sequence[Any] | None) -> str:
    """Lowercased text corpus from upstream artifacts (excludes final_report)."""
    chunks: list[str] = []
    for artifact in filter_evidence_artifacts(artifacts):
        content = _get(artifact, "content")
        if isinstance(content, dict):
            chunks.append(json.dumps(content))
        elif content is not None:
            chunks.append(str(content))
    return "\n".join(chunks).lower()
