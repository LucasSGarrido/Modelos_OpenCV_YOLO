# Dados

Esta pasta guarda apenas amostras pequenas e documentação. Não coloque datasets completos ou vídeos pesados no GitHub.

## Estrutura

- `samples/`: imagens ou vídeos curtos para teste local.
- `annotations/`: arquivos manuais de contagem, quando existirem.
- `sources/`: fontes, autores, licenças e observações das amostras reais.

## Ficha do Dataset

Domínio definido para esta versão: **pessoas e veículos**.

Estado atual:

- Nome: amostra pequena de pessoas/veículos para validação inicial
- Fonte: Ultralytics assets + Wikimedia Commons
- Link: ver `sources/real_samples.csv`
- Autor/organização: ver `sources/real_samples.csv`
- Data de acesso: 2026-05-11 para as amostras Wikimedia
- Licença: public domain para as duas imagens reais adicionadas
- Tipo de mídia: imagem e vídeo curto sintético
- Classes de interesse: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`
- Quantidade de imagens reais adicionadas: 2
- Quantidade de vídeos reais adicionados: 0
- Existe anotação manual? Sim, em `annotations/counts.csv`
- Observações de privacidade: as imagens reais são cenas urbanas públicas de Wikimedia Commons.

## Amostras Reais

- `samples/real_traffic_bus_france.jpg`: congestionamento com ônibus e carros na França.
- `samples/real_intersection_auckland.jpg`: cruzamento urbano em Auckland.
- Metadados: `sources/real_samples.csv`.
- Artefatos processados: `reports/real/`.
- Observação: a anotação atual dessas imagens foca a classe `bus`, porque a contagem de carros/persons na cena ampla exigiria uma rodada manual mais detalhada.

## Amostra Local Atual

- Arquivo: `samples/bus.jpg`
- Origem: assets de exemplo instalados com o pacote `ultralytics`
- Uso: teste local do pipeline e geração de primeira imagem processada
- Observação: essa amostra serve apenas para demonstração técnica inicial; o dataset final do projeto ainda precisa ser escolhido e documentado.

## Vídeo de Demonstração Local

- Arquivo: `samples/bus_demo.mp4`
- Como gerar: `python scripts/create_demo_video.py`
- Origem: vídeo sintético criado a partir de `samples/bus.jpg`
- Uso: validar o modo de contagem por linha sem depender de um dataset externo
- Observação: não deve ser usado para afirmar métricas reais de acurácia.

## Anotações Manuais

- Arquivo atual: `annotations/counts.csv`
- Uso: comparar contagens esperadas contra os JSONs gerados pelo pipeline
- Como avaliar: `python scripts/evaluate_counts.py`
- Saídas: `reports/evaluation/counts_report.csv`, `reports/evaluation/counts_summary.json` e `reports/evaluation/error_analysis.csv`
- Como otimizar thresholds: `python scripts/optimize_thresholds.py`
- Saídas da otimização: `reports/evaluation/threshold_search.csv` e `reports/evaluation/best_thresholds.json`
- Como comparar modelos: `python scripts/compare_models.py --models yolov8n.pt yolov8s.pt`
- Saídas da comparação: `reports/evaluation/model_comparison.csv` e `reports/evaluation/best_model.json`

## Regras

- Evite imagens com pessoas identificáveis sem autorização.
- Não versionar arquivos grandes.
- Registre sempre fonte, licença e data de acesso.
- Se usar vídeos próprios, remova informações sensíveis antes de publicar.
