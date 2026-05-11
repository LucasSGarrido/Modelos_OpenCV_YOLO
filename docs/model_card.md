# Model Card

## Modelo

- Tipo: YOLO pré-treinado via Ultralytics
- Peso padrão: `yolov8n.pt`
- Modelo comparado: `yolov8s.pt`
- Peso padrão de segmentação: `yolov8n-seg.pt`
- Peso padrão de pose/keypoints: `yolov8n-pose.pt`
- Fine-tuning: não realizado nesta fase

## Uso Pretendido

Detectar objetos comuns em imagens e vídeos curtos para gerar contagem por classe e visualização com bounding boxes.

Na versão atual, o modelo YOLO pode ser combinado com tracker por centroide ou ByteTrack para permitir contagem por cruzamento de linha em vídeos curtos.

O projeto também inclui páginas separadas para:

- segmentação de instâncias em imagens e vídeos curtos, com máscaras, polígonos e área aproximada por objeto;
- keypoint/pose detection em imagens e vídeos curtos, com pontos corporais e esqueleto para pessoas.

## Fora do Escopo

- Identificação individual de pessoas.
- Reconhecimento facial.
- Uso como sistema crítico de segurança.
- Afirmações de acurácia sem avaliação manual.
- Tracking e contagem de eventos nas páginas de segmentação e pose nesta fase.

## Métricas Planejadas

- Precision e recall, se houver anotação de bounding boxes.
- Erro absoluto de contagem.
- MAE de contagem.
- Falsos positivos e falsos negativos.
- FPS médio.
- Total de eventos por linha, quando o modo `line` for usado.
- Comparação entre modelos por erro de contagem e tempo por amostra.
- Área total, área por classe, maior máscara e percentual mascarado em segmentação.
- Total de pessoas, keypoints visíveis e confiança média dos pontos em pose.
- Avaliação manual de segmentação e pose por `data/annotations/segmentation_pose.csv`.
- Diagnóstico comentado para revisar falhas prováveis antes de interpretar métricas.
- CSV por frame para vídeos de segmentação e pose.

## Resultado Atual

- Melhor threshold nas amostras atuais ampliadas: `confidence=0.25`, `iou=0.70`.
- Comparação atual: `yolov8n.pt` ficou melhor que `yolov8s.pt` no conjunto pequeno anotado.
- Interpretação: o resultado ainda não prova superioridade geral do modelo nano; apenas mostra que, neste conjunto pequeno e focado em ônibus, ele teve melhor equilíbrio entre erro e tempo.

## Limitações

- Pode falhar em objetos pequenos.
- Pode confundir classes visualmente parecidas.
- Pode perder objetos parcialmente ocultos.
- Pode ter desempenho baixo em CPU dependendo da resolução.
- Trackers podem trocar ou perder IDs quando objetos se cruzam, ficam ocultos, saem da ROI ou se movem rápido.
- ByteTrack tende a ser mais robusto que o centroide em cenas difíceis, mas ainda precisa ser validado em vídeos reais do domínio.
- Segmentação e pose usam pesos especializados; `yolov8n.pt` não gera máscaras nem keypoints.
- Segmentação e pose já possuem avaliação manual inicial, mas ainda precisam de mais amostras antes de qualquer afirmação forte de qualidade.
- A interface bloqueia pesos sem sufixo esperado nas páginas especializadas: `-seg.pt` para segmentação e `-pose.pt` para pose.
- URLs de entrada podem apontar para arquivos de mídia, YouTube, Google Drive público ou Dropbox. O processamento deve respeitar direitos de uso, privacidade e restrições da plataforma.
