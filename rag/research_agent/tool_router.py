"""Allowlisted research tool router with a private workspace path boundary.

Designed for future LLM-tool integration. This module itself does not parse
LLM output, execute shell commands or expose HTTP endpoints. The caller must
set a trusted workspace root and apply authentication at any API boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.research_tools.registry import run_named_tool


class ToolSecurityError(ValueError):
    pass


ALLOWED_TOOL_NAMES = frozenset({"spectroscopy.report", "presentation.from_discussion"})
SPECTRUM_EXTENSIONS = frozenset({".csv", ".tsv", ".txt", ".asc", ".dat"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
MAX_SPECTRUM_BYTES = 64 * 1024 * 1024
MAX_TEMPLATE_BYTES = 32 * 1024 * 1024


def _within_workspace(workspace: Path, value: str | Path, must_exist: bool) -> Path:
    target = Path(value).expanduser()
    if not target.is_absolute():
        target = workspace / target
    target = target.resolve(strict=must_exist)
    if not target.is_relative_to(workspace):
        raise ToolSecurityError("Path outside approved research workspace")
    return target


class ResearchToolRouter:
    def __init__(self, workspace_root: str | Path):
        root = Path(workspace_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ToolSecurityError("workspace_root must be an existing directory")
        self.root = root

    def _input(self, path: str | Path, suffixes: frozenset[str], max_size: int) -> Path:
        f = _within_workspace(self.root, path, must_exist=True)
        if not f.is_file() or f.suffix.lower() not in suffixes:
            raise ToolSecurityError("File type or input path not allowed")
        if f.stat().st_size > max_size:
            raise ToolSecurityError("Input exceeds maximum permitted file size")
        return f

    def _output(self, path: str | Path, suffix: str | None = None) -> Path:
        out = _within_workspace(self.root, path, must_exist=False)
        if suffix and out.suffix.lower() != suffix:
            raise ToolSecurityError(f"Output must use {suffix} extension")
        # Resolve symlinked parent directories, even when the output is new.
        if not out.parent.resolve(strict=False).is_relative_to(self.root):
            raise ToolSecurityError("Output directory escapes workspace")
        return out

    def run(self, tool_name: str, arguments: dict[str, Any]) -> dict:
        if tool_name not in ALLOWED_TOOL_NAMES:
            raise ToolSecurityError("Tool name not in allowlist")
        if not isinstance(arguments, dict):
            raise ToolSecurityError("Tool arguments must be an object")
        args = dict(arguments)
        if tool_name == "spectroscopy.report":
            allowed = {"measurements", "output_dir", "deck_title", "template_path", "create_pptx"}
            if set(args) - allowed:
                raise ToolSecurityError("Unexpected spectroscopy tool argument")
            samples = args.get("measurements")
            if not isinstance(samples, list) or not 1 <= len(samples) <= 32:
                raise ToolSecurityError("Need 1-32 measurements")
            normalized = []
            sample_keys = {"path", "technique", "sample_id", "signal_mode",
                           "min_prominence", "min_distance_points", "measurement_id", "x_unit"}
            for item in samples:
                if not isinstance(item, dict) or set(item) - sample_keys:
                    raise ToolSecurityError("Invalid measurement fields")
                row = dict(item)
                try:
                    source = self._input(row["path"], SPECTRUM_EXTENSIONS, MAX_SPECTRUM_BYTES)
                    row["sample_id"] = str(row["sample_id"])
                    row["technique"] = str(row["technique"])
                    row["min_prominence"] = float(row["min_prominence"])
                except (KeyError, TypeError, ValueError) as err:
                    raise ToolSecurityError("Invalid measurement parameters") from err
                row["path"] = str(source)
                normalized.append(row)
            args["measurements"] = normalized
            args["output_dir"] = str(self._output(args.get("output_dir", "results")))
        elif tool_name == "presentation.from_discussion":
            allowed = {"title", "findings", "next_steps", "unresolved_questions",
                       "output_path", "template_path"}
            if set(args) - allowed:
                raise ToolSecurityError("Unexpected presentation tool argument")
            if not isinstance(args.get("findings"), list) or not 1 <= len(args["findings"]) <= 40:
                raise ToolSecurityError("Need 1-40 findings")
            for finding in args["findings"]:
                if not isinstance(finding, dict):
                    raise ToolSecurityError("Invalid finding")
                if len(finding.get("figure_paths", [])) > 2:
                    raise ToolSecurityError("At most two figures per slide")
                finding["figure_paths"] = [
                    str(self._input(p, IMAGE_EXTENSIONS, MAX_SPECTRUM_BYTES))
                    for p in finding.get("figure_paths", ())
                ]
            args["output_path"] = str(self._output(args.get("output_path", "results/report.pptx"), ".pptx"))
        if args.get("template_path"):
            args["template_path"] = str(
                self._input(args["template_path"], frozenset({".pptx"}), MAX_TEMPLATE_BYTES)
            )
        return run_named_tool(tool_name, args)
