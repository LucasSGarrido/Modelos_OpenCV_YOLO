from pathlib import Path

import pytest

from object_counter.utils.io import default_output_path, infer_media_type


def test_infer_media_type_for_images_and_videos() -> None:
    assert infer_media_type("foto.jpg") == "image"
    assert infer_media_type("video.mp4") == "video"


def test_infer_media_type_rejects_unknown_extension() -> None:
    with pytest.raises(ValueError):
        infer_media_type("arquivo.txt")


def test_default_output_path_keeps_expected_folder() -> None:
    assert default_output_path("entrada/foto.png", "image") == Path("reports/figures/foto_processado.jpg")
    assert default_output_path("entrada/video.mov", "video") == Path("reports/videos/video_processado.mp4")
