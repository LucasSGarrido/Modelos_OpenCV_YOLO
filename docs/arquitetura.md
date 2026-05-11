# Arquitetura

O projeto foi dividido em módulos pequenos para facilitar evolução e explicação em entrevista.

## Fluxo

```text
src/main.py
↓
utils/io.py identifica imagem ou vídeo
↓
detection/detector.py executa YOLO
↓
tracking/ associa IDs por centroide ou ByteTrack em vídeos
↓
counting/ conta objetos por classe ou eventos de cruzamento
↓
visualization/draw.py gera overlay visual
↓
utils/reports.py exporta relatórios Markdown/HTML
↓
evaluation/ apoia avaliação manual, análise de erros, diagnósticos e busca de thresholds
↓
scripts/compare_models.py compara modelos YOLO usando as mesmas anotações
```

## Páginas da Interface

```text
src/app.py
|
+-- app_pages/instance_segmentation.py -> segmentation/segmenter.py -> visualization/segmentation.py
+-- app_pages/keypoint_detection.py    -> pose/pose_detector.py       -> visualization/pose.py
+-- app_pages/custom_training.py       -> training/guidance.py
+-- fluxo legado de detecção/contagem  -> detection/ + counting/ + tracking/
```

As novas páginas foram adicionadas sem desmontar o fluxo principal de contagem. Segmentação e pose aceitam imagem e vídeo curto via upload, amostra local, URL direta de arquivo, URL do YouTube, Google Drive público ou Dropbox, validam o tipo de peso esperado, permitem ROI por sliders e mostram progresso durante vídeos. A página de treinamento customizado orienta marcação, volume de dados, estrutura YOLO, `data.yaml` e comando de treino. A lógica de tracking/eventos continua concentrada no fluxo principal de detecção e contagem.

URLs remotas passam por `utils/downloads.py`, que valida protocolo `http/https`, extensão suportada, nome de arquivo e limite máximo de download antes de salvar a mídia temporariamente para o pipeline local. URLs do YouTube são resolvidas com `yt-dlp`; Google Drive usa o endpoint público de download e Dropbox troca links compartilhados para `dl=1` quando possível.

Segmentação registra métricas específicas no JSON: área total de máscaras, percentual mascarado, maior máscara, classe da maior máscara e área por classe. Para vídeos, o resumo guarda máximos observados e o CSV inclui razão de área por frame.

A avaliação especializada usa `data/annotations/segmentation_pose.csv` e `scripts/evaluate_specialized_tasks.py`. O módulo `evaluation/task_report.py` compara métricas esperadas contra resumos JSON, enquanto `evaluation/vision_diagnostics.py` gera comentários automáticos para revisar falhas prováveis em segmentação e pose.

O módulo `training/guidance.py` mantém a parte testável da página de treinamento: parse de classes, estimativa de volume mínimo/recomendado, geração de `data.yaml` e comando `yolo ... train`. A interface não dispara treino longo dentro do Streamlit; ela prepara a pessoa para rodar o treino em ambiente adequado.

## Decisões

- YOLO pré-treinado foi escolhido como baseline para reduzir tempo inicial de implementação.
- A contagem em imagem é direta: cada detecção válida conta uma vez.
- A contagem em vídeo possui dois modos: `frame`, para contagem instantânea, e `line`, para eventos de cruzamento.
- O tracking por centroide é simples e explicável, bom para demos curtas.
- ByteTrack está disponível como alternativa mais robusta para o modo `line`.
- DeepSORT continua como candidato futuro se as cenas reais exigirem reidentificação mais forte.
- Métricas de acurácia só devem ser publicadas depois de anotar uma amostra de validação.
- Batch de imagens foi separado em `scripts/process_image_batch.py` para não complicar o fluxo principal da interface.
- A otimização de `confidence`/`iou` foi separada em `scripts/optimize_thresholds.py` e usa as anotações manuais como referência.
- A comparação de modelos foi separada em `scripts/compare_models.py`; no conjunto atual, `yolov8n.pt` teve menor erro que `yolov8s.pt`.
- A avaliação de segmentação e pose foi separada do avaliador de contagem para aceitar métricas numéricas além de classes, como área de máscara e keypoints visíveis.
- O treinamento customizado foi tratado como planejamento e geração de artefatos/comandos, porque treino de YOLO pode ser longo e deve rodar fora do ciclo interativo principal.
- A ROI desenhável em vídeo usa o primeiro frame como superfície de desenho para manter a interação simples.
- Segmentação usa modelos `*-seg.pt` porque modelos de detecção como `yolov8n.pt` não retornam máscaras.
- Pose usa modelos `*-pose.pt` porque modelos de detecção comuns não retornam keypoints.
- Segmentação e pose foram isoladas em módulos próprios para permitir evolução para vídeo sem acoplar ao contador principal.
