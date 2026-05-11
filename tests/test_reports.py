import json

from object_counter.utils.reports import (
    build_comparison_html,
    build_comparison_markdown,
    build_run_html,
    build_run_markdown,
    compare_history_records,
    counts_from_history_record,
)


def test_counts_from_history_record_reads_summary(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"counts": {"person": 2}}), encoding="utf-8")

    counts = counts_from_history_record(
        {
            "summary_path": str(summary_path),
            "media_type": "image",
            "mode": "image",
        }
    )

    assert counts == {"person": 2}


def test_compare_history_records_computes_class_difference(tmp_path) -> None:
    left_summary = tmp_path / "left.json"
    right_summary = tmp_path / "right.json"
    left_summary.write_text(json.dumps({"counts": {"person": 2, "car": 1}}), encoding="utf-8")
    right_summary.write_text(json.dumps({"counts": {"person": 3, "car": 1}}), encoding="utf-8")

    comparison = compare_history_records(
        {"summary_path": str(left_summary), "media_type": "image", "mode": "image", "total": 3},
        {"summary_path": str(right_summary), "media_type": "image", "mode": "image", "total": 4},
    )

    assert {"classe": "person", "execução_a": 2, "execução_b": 3, "diferença": 1} in comparison[
        "classes"
    ]


def test_markdown_reports_have_expected_titles() -> None:
    run_report = build_run_markdown({"counts": {"person": 1}, "total": 1}, media_type="image")
    comparison_report = build_comparison_markdown(
        {"input_name": "a.jpg"},
        {"input_name": "b.jpg"},
        {"overview": [], "classes": []},
    )

    assert "# Relatório da Execução" in run_report
    assert "# Relatório de Comparação de Execuções" in comparison_report


def test_html_reports_escape_content() -> None:
    run_report = build_run_html({"counts": {"person": 1}, "input_path": "<x>"}, media_type="image")
    comparison_report = build_comparison_html(
        {"input_name": "a.jpg"},
        {"input_name": "b.jpg"},
        {"overview": [], "classes": []},
    )

    assert "&lt;x&gt;" in run_report
    assert "<title>Relatório de Comparação de Execuções</title>" in comparison_report
