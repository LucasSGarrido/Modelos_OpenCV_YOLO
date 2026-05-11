from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avalia métricas manuais de segmentação e pose contra resumos JSON."
    )
    parser.add_argument("--annotations", default="data/annotations/segmentation_pose.csv")
    parser.add_argument("--output", default="reports/evaluation/segmentation_pose_report.csv")
    parser.add_argument(
        "--summary-output",
        default="reports/evaluation/segmentation_pose_summary.json",
    )
    parser.add_argument(
        "--error-output",
        default="reports/evaluation/segmentation_pose_error_analysis.csv",
    )
    return parser


def main() -> int:
    from object_counter.evaluation.task_report import (
        evaluate_task_annotations,
        write_task_evaluation_report,
    )

    args = build_parser().parse_args()
    report = evaluate_task_annotations(args.annotations)
    write_task_evaluation_report(report, args.output, args.summary_output, args.error_output)
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
