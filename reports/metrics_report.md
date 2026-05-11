# Relatório de Métricas

Ainda não há resultados reais calculados. Este arquivo deve ser preenchido depois que uma amostra for anotada manualmente.

## Tabela Esperada

| Amostra | Classe | Real | Previsto | Erro absoluto | Observação |
| --- | --- | ---: | ---: | ---: | --- |
| `bus_image` | `bus` | 1 | 1 | 0 | Amostra visual local |
| `bus_image` | `person` | 3 | 3 | 0 | Amostra visual local |
| `bus_demo_line` | `bus` | 1 | 1 | 0 | Demo sintética em modo linha |
| `bus_demo_line` | `person` | 3 | 3 | 0 | Demo sintética em modo linha |

## Resumo Atual

| Métrica | Valor |
| --- | ---: |
| Amostras avaliadas | 2 |
| Linhas por classe | 4 |
| Erro absoluto total | 0 |
| MAE de contagem | 0.0 |
| MAPE de contagem | 0.0 |
| Taxa de acerto exato | 100% |

## Performance

| Arquivo | Tipo | Frames processados | FPS médio | Observação |
| --- | --- | ---: | ---: | --- |
| `bus_demo.mp4` | demo sintética | 72 | 33.4 | Modo linha, CPU/local |

## Análise de Falhas

- Falsos positivos:
- Falsos negativos:
- Baixa iluminação:
- Objetos pequenos:
- Oclusão:

## Demo Técnica

| Arquivo | Modo | Contagem observada | Observação |
| --- | --- | --- | --- |
| `bus.jpg` | imagem | `1 bus`, `3 person` | Amostra local do Ultralytics |
| `bus_demo.mp4` | linha | `1 bus`, `3 person` | Vídeo sintético criado a partir de `bus.jpg`; não representa métrica real de acurácia |
| `bus.jpg` | imagem com ROI | `1 bus`, `2 person` | ROI relativa `0.0 0.30 0.55 1.0`; filtra pelo centro da bounding box |
