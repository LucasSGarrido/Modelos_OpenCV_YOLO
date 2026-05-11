from __future__ import annotations

from typing import Any


def segmentation_diagnostic_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build human-readable diagnostic rows for segmentation summaries."""
    is_video = str(summary.get("model_task", "")).endswith("_video")
    total = int(summary.get("max_frame_total", 0) if is_video else summary.get("total", 0))
    rows: list[dict[str, Any]] = []

    if total == 0:
        rows.append(
            _row(
                "alta",
                "sem_mascaras",
                0,
                "Nenhuma máscara foi detectada no resultado.",
                "Reduza a confiança mínima, revise as classes selecionadas e confirme se o peso termina em -seg.pt.",
            )
        )
    else:
        rows.append(
            _row(
                "baixa",
                "mascaras_detectadas",
                total,
                "A segmentação encontrou instâncias para revisar.",
                "Compare as máscaras com a cena original e confirme se objetos parcialmente visíveis devem contar.",
            )
        )

    area_metrics = summary.get("last_frame_area_metrics", {}) if is_video else summary.get("area_metrics", {})
    area_ratio = _number(
        summary.get("max_frame_area_ratio") if is_video else area_metrics.get("mask_area_ratio")
    )
    if total > 0 and area_ratio <= 0.01:
        rows.append(
            _row(
                "média",
                "area_muito_pequena",
                round(area_ratio, 6),
                "As máscaras ocupam uma área muito pequena da cena.",
                "Verifique objetos distantes, resolução baixa, ROI apertada ou limite de confiança alto.",
            )
        )
    elif total > 0 and area_ratio >= 0.65:
        rows.append(
            _row(
                "média",
                "area_muito_grande",
                round(area_ratio, 6),
                "As máscaras ocupam grande parte da cena.",
                "Confira se o modelo está cobrindo fundo, reflexos ou objetos colados na borda.",
            )
        )

    largest_area = _number(summary.get("largest_mask_area") or area_metrics.get("largest_mask_area"))
    average_area = _number(area_metrics.get("average_mask_area"))
    if total > 1 and average_area > 0 and largest_area / average_area >= 4:
        rows.append(
            _row(
                "média",
                "mascara_dominante",
                round(largest_area / average_area, 2),
                "Uma máscara é muito maior que a média das demais.",
                "Revise se houve união de instâncias próximas ou se a maior máscara representa fundo.",
            )
        )

    counts = summary.get("max_counts_by_class", {}) if is_video else summary.get("counts", {})
    if len(counts) > 3:
        rows.append(
            _row(
                "baixa",
                "muitas_classes",
                len(counts),
                "O resultado mistura várias classes.",
                "Use o filtro de classes quando quiser medir um grupo específico.",
            )
        )

    return rows or [
        _row(
            "baixa",
            "sem_alertas",
            "-",
            "Nenhum alerta automático relevante foi encontrado.",
            "Use a avaliação manual para confirmar qualidade por amostra.",
        )
    ]


def pose_diagnostic_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build human-readable diagnostic rows for pose/keypoint summaries."""
    is_video = str(summary.get("model_task", "")).endswith("_video")
    people = int(summary.get("max_people", 0) if is_video else summary.get("total", 0))
    visible_keypoints = int(
        summary.get("max_visible_keypoints", 0)
        if is_video
        else summary.get("visible_keypoints", 0)
    )
    average_confidence = summary.get("average_keypoint_confidence")
    rows: list[dict[str, Any]] = []

    if people == 0:
        rows.append(
            _row(
                "alta",
                "sem_pessoas",
                0,
                "Nenhuma pessoa foi detectada para estimar pose.",
                "Revise a cena, a ROI e a confiança mínima do detector de pose.",
            )
        )
    else:
        keypoints_per_person = visible_keypoints / people if people else 0.0
        rows.append(
            _row(
                "baixa",
                "pessoas_detectadas",
                people,
                "O modelo encontrou pessoas e pontos corporais para revisar.",
                "Confira visualmente se a estrutura acompanha cabeça, tronco e membros.",
            )
        )
        if keypoints_per_person < 8:
            rows.append(
                _row(
                    "média",
                    "poucos_keypoints",
                    round(keypoints_per_person, 2),
                    "Há poucos keypoints visíveis por pessoa.",
                    "Oclusão, distância, cortes no corpo ou baixa luz podem estar reduzindo a pose.",
                )
            )

    if average_confidence is not None and float(average_confidence) < 0.45:
        rows.append(
            _row(
                "média",
                "confianca_baixa",
                round(float(average_confidence), 4),
                "A confiança média dos keypoints está baixa.",
                "Aumente a qualidade da imagem ou ajuste ROI/thresholds antes de concluir a avaliação.",
            )
        )

    if is_video and int(summary.get("frames_processed", 0)) == 0:
        rows.append(
            _row(
                "alta",
                "video_sem_frames",
                0,
                "Nenhum frame foi processado no vídeo.",
                "Confirme codec, caminho do arquivo, limite de frames e permissão de leitura.",
            )
        )

    return rows or [
        _row(
            "baixa",
            "sem_alertas",
            "-",
            "Nenhum alerta automático relevante foi encontrado.",
            "Use a avaliação manual para confirmar qualidade por amostra.",
        )
    ]


def _row(
    severity: str,
    indicator: str,
    value: Any,
    comment: str,
    suggested_action: str,
) -> dict[str, Any]:
    return {
        "severidade": severity,
        "indicador": indicator,
        "valor": value,
        "comentário": comment,
        "ação_sugerida": suggested_action,
    }


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
