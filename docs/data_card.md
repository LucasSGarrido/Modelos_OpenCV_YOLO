# Data Card

## Status

Dataset final ainda não definido. O domínio de validação foi definido como **pessoas e veículos**, usando YOLO COCO como baseline. A versão atual já inclui duas imagens reais abertas do Wikimedia Commons para uma avaliação inicial.

## Fonte

- Nome: amostras urbanas de pessoas/veículos
- URL: ver `data/sources/real_samples.csv`
- Autor/organização: Cartedd e Ingolfson
- Data de acesso: 2026-05-11
- Licença: public domain nas duas imagens reais adicionadas

## Cobertura

- Classes de interesse: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`
- Tipo de mídia: imagens reais e vídeo sintético curto de demonstração
- Quantidade de imagens reais: 2
- Quantidade de vídeos reais: 0
- Duração total: não aplicável para imagens reais
- Resolução: varia por arquivo

## Qualidade

- Existe anotação manual? Sim, em `data/annotations/counts.csv`.
- Há baixa iluminação? Não nas amostras atuais.
- Há oclusão? Sim, especialmente em `real_traffic_bus_france.jpg`.
- Há objetos pequenos? Sim, ônibus distantes nas duas imagens reais.
- Há pessoas identificáveis? Não é o foco das anotações reais atuais; cenas são públicas e amplas.

## Limitações

- O conjunto real ainda é pequeno e focado em ônibus.
- A contagem de carros nas cenas amplas ainda não foi anotada manualmente.
- As métricas atuais são úteis para validar pipeline e erro, mas não representam performance final em produção.
