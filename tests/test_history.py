from pathlib import Path

import pandas as pd

from object_counter.utils.history import (
    append_history_csv,
    build_history_record,
    filter_history,
    load_history_csv,
)


def test_build_history_record_for_image() -> None:
    record = build_history_record(
        result={
            "input_path": "data/samples/bus.jpg",
            "total": 4,
            "output_path": "out.jpg",
            "summary_path": "out.json",
        },
        media_type="image",
        model_config={"classes": ["bus", "person"], "confidence": 0.35, "iou": 0.5},
        video_config={"roi_enabled": True},
    )

    assert record["input_name"] == "bus.jpg"
    assert record["mode"] == "image"
    assert record["total"] == 4
    assert record["classes"] == "bus person"
    assert record["roi_enabled"] is True


def test_append_and_load_history_csv(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    append_history_csv(path, {"input_name": "a.jpg", "media_type": "image", "mode": "image"})

    history = load_history_csv(path)

    assert len(history) == 1
    assert history.loc[0, "input_name"] == "a.jpg"


def test_filter_history() -> None:
    history = pd.DataFrame(
        [
            {"input_name": "bus.jpg", "media_type": "image", "mode": "image", "classes": "bus"},
            {"input_name": "video.mp4", "media_type": "video", "mode": "line", "classes": "person"},
        ]
    )

    filtered = filter_history(history, media_type="video", mode="line", search="person")

    assert filtered["input_name"].tolist() == ["video.mp4"]
