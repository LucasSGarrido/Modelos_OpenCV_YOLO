from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from object_counter.evaluation.count_report import predicted_counts_from_summary


def history_record_label(record: dict[str, Any], index: int) -> str:
    input_name = record.get("input_name", "arquivo")
    run_at = record.get("run_at", "-")
    mode = record.get("mode", "-")
    total = record.get("total", "-")
    return f"{index} · {input_name} · {mode} · total {total} · {run_at}"


def counts_from_history_record(record: dict[str, Any]) -> dict[str, int]:
    summary_path = str(record.get("summary_path", "") or "")
    if not summary_path or not Path(summary_path).exists():
        return {}

    with Path(summary_path).open("r", encoding="utf-8") as file:
        summary = json.load(file)

    return predicted_counts_from_summary(
        summary=summary,
        media_type=str(record.get("media_type", "")),
        counting_mode=str(record.get("mode", "")),
    )


def compare_history_records(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_counts = counts_from_history_record(left)
    right_counts = counts_from_history_record(right)
    labels = sorted(set(left_counts) | set(right_counts))
    class_rows = [
        {
            "classe": label,
            "execução_a": left_counts.get(label, 0),
            "execução_b": right_counts.get(label, 0),
            "diferença": right_counts.get(label, 0) - left_counts.get(label, 0),
        }
        for label in labels
    ]

    total_left = int(left.get("total", 0) or 0)
    total_right = int(right.get("total", 0) or 0)
    overview_rows = [
        {"métrica": "arquivo", "execução_a": left.get("input_name", ""), "execução_b": right.get("input_name", "")},
        {"métrica": "tipo", "execução_a": left.get("media_type", ""), "execução_b": right.get("media_type", "")},
        {"métrica": "modo", "execução_a": left.get("mode", ""), "execução_b": right.get("mode", "")},
        {"métrica": "classes", "execução_a": left.get("classes", ""), "execução_b": right.get("classes", "")},
        {"métrica": "total", "execução_a": total_left, "execução_b": total_right},
        {"métrica": "diferença total", "execução_a": "", "execução_b": total_right - total_left},
        {"métrica": "FPS", "execução_a": left.get("fps", ""), "execução_b": right.get("fps", "")},
        {"métrica": "tempo", "execução_a": left.get("processing_seconds", ""), "execução_b": right.get("processing_seconds", "")},
        {"métrica": "ROI", "execução_a": left.get("roi_enabled", ""), "execução_b": right.get("roi_enabled", "")},
        {"métrica": "tracking", "execução_a": left.get("tracking_backend", ""), "execução_b": right.get("tracking_backend", "")},
        {"métrica": "confiança", "execução_a": left.get("confidence", ""), "execução_b": right.get("confidence", "")},
        {"métrica": "IOU", "execução_a": left.get("iou", ""), "execução_b": right.get("iou", "")},
        {"métrica": "artefato", "execução_a": left.get("output_path", ""), "execução_b": right.get("output_path", "")},
    ]

    return {
        "overview": overview_rows,
        "classes": class_rows,
        "left_counts": left_counts,
        "right_counts": right_counts,
    }


def build_comparison_markdown(
    left: dict[str, Any],
    right: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    overview = markdown_table(comparison["overview"], ["métrica", "execução_a", "execução_b"])
    classes = markdown_table(
        comparison["classes"],
        ["classe", "execução_a", "execução_b", "diferença"],
    )
    return (
        "# Relatório de Comparação de Execuções\n\n"
        "## Execução A\n\n"
        f"- Arquivo: `{left.get('input_name', '')}`\n"
        f"- Data/hora: `{left.get('run_at', '')}`\n"
        f"- Saída: `{left.get('output_path', '')}`\n\n"
        "## Execução B\n\n"
        f"- Arquivo: `{right.get('input_name', '')}`\n"
        f"- Data/hora: `{right.get('run_at', '')}`\n"
        f"- Saída: `{right.get('output_path', '')}`\n\n"
        "## Resumo\n\n"
        f"{overview}\n\n"
        "## Contagem por Classe\n\n"
        f"{classes if comparison['classes'] else 'Sem contagens por classe disponíveis.'}\n"
    )


def build_comparison_html(
    left: dict[str, Any],
    right: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    return markdown_to_html_document(
        title="Relatório de Comparação de Execuções",
        markdown=build_comparison_markdown(left, right, comparison),
    )


def build_run_markdown(result: dict[str, Any], media_type: str | None) -> str:
    if media_type == "image":
        counts = result.get("counts", {})
        total = result.get("total", 0)
        timing_label = "Inferência"
        timing_value = f"{float(result.get('inference_seconds', 0)):.4f}s"
    else:
        counts = (
            result.get("line_counts_by_class", {})
            if result.get("counting_mode") == "line"
            else result.get("last_frame_counts", {})
        )
        total = result.get("total_line_crossings", sum(counts.values()))
        timing_label = "FPS médio"
        timing_value = f"{float(result.get('average_processing_fps', 0)):.2f}"

    counts_rows = [{"classe": label, "contagem": value} for label, value in sorted(counts.items())]
    counts_markdown = (
        markdown_table(counts_rows, ["classe", "contagem"]) if counts_rows else "Sem detecções."
    )

    return (
        "# Relatório da Execução\n\n"
        f"- Tipo: `{media_type}`\n"
        f"- Modo: `{result.get('counting_mode', 'image')}`\n"
        f"- Total: `{total}`\n"
        f"- {timing_label}: `{timing_value}`\n"
        f"- Entrada: `{result.get('input_path', '')}`\n"
        f"- Saída: `{result.get('output_path', '')}`\n"
        f"- Resumo: `{result.get('summary_path', '')}`\n\n"
        "## Contagem por Classe\n\n"
        f"{counts_markdown}\n"
    )


def build_run_html(result: dict[str, Any], media_type: str | None) -> str:
    return markdown_to_html_document(
        title="Relatório da Execução",
        markdown=build_run_markdown(result, media_type),
    )


def markdown_to_html_document(title: str, markdown: str) -> str:
    escaped = (
        markdown.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        "<!doctype html><html lang=\"pt-BR\"><head><meta charset=\"utf-8\">"
        f"<title>{title}</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:960px;margin:40px auto;"
        "line-height:1.5;color:#111827}pre{white-space:pre-wrap;background:#f3f4f6;"
        "padding:16px;border-radius:8px}</style></head><body>"
        f"<pre>{escaped}</pre></body></html>"
    )


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])
