# rec_pad — Ensemble evolucionário (GA) para previsão de séries temporais

Projeto da disciplina **Reconhecimento de Padrões** (RECPAD · PPGEC · 2026.1).
Comparamos vários regressores clássicos na previsão **um-passo-à-frente** de séries
temporais univariadas e os combinamos com um **Algoritmo Genético (GA)** que aprende
os pesos de mesclagem por modelo.

Base de referência: Barchi, T. M., dos Santos, J. L. F., Alves, T. A., *et al.*
(2026). *A genetic algorithm-based ensemble framework for wind speed forecasting.*
**Scientific Reports, 16, 6847.** doi:10.1038/s41598-026-37003-1.

## Problema

Estimar `yₜ` a partir de uma janela dos `p` valores anteriores
`[y₍ₜ₋ₚ₎ … y₍ₜ₋₁₎]` (`n_lags = 5`). Nenhum modelo único vence em todas as séries —
o melhor muda por base e o KNN é consistentemente o pior. A proposta combina
aprendizes heterogêneos com pesos aprendidos por GA no **simplex** (`w ≥ 0`,
`Σw = 1`), reduzindo o risco de escolher o modelo errado para cada série.

**Contribuições sobre o artigo base:** (i) etapa de **seleção TOP-4** por erro de
validação antes do GA (o artigo combina todos os modelos); (ii) mudança de domínio
(do vento para séries clássicas + bases UCI) com pool heterogêneo implementado do
zero (Adaline, ELM, SCN, KNN) — atende ao requisito de "não usar apenas sklearn".

## Bases de dados

Séries univariadas, uma coluna com cabeçalho `y` (carregar com
`pd.read_csv(path)`, `header=0`):

| Base | Domínio | Origem |
|------|---------|--------|
| `airlines` | passageiros aéreos (mensal) | `data/airlines.txt` |
| `lynx` | capturas de lince (anual) | `data/lynx.txt` |
| `coloradoRiver` | vazão do rio (mensal) | `data/coloradoRiver.txt` |
| `melbmin` | temp. mínima diária (Melbourne) | `data/melbmin.txt` |
| `air_quality_rh` | umidade relativa do ar (horária) | UCI 360, via `ucimlrepo` |
| `appliances_energy` | consumo de energia elétrica | UCI 374, via `ucimlrepo` |

## Estrutura do repositório

```
conf/
  base_model.py      # BaseModel: estado de seleção do melhor candidato
  processing.py      # DataProcessing: janelamento, split temporal, normalização
models/
  adaline.py         # Adaline (do zero)
  ann_models.py      # MLP, ELM (GridELM), SCN (GridSCN)
  k_near.py          # KNEAR (KNN, busca de vizinhos própria)
  kernel_method.py   # SVM (SVR do sklearn)
  tree_based_ensemble.py  # RF, GBoosting (disponíveis; fora do pool do experimento de 6 bases)
utils/
  genetic_algorithm.py    # GeneticAlgorithm: pesos do ensemble no simplex
  plot_dashes.py          # métricas (get_metrics_error) e gráficos (PT-BR)
data/                # séries clássicas (.txt)
benchmark.py         # tabela MSE/RMSE/MAE/MAPE de todos os modelos × bases
dataset_forecast.ipynb        # experimento por base
global_metrics_forecast.ipynb # experimento global nas 6 bases (gera results/)
presentation/        # deck RECPAD (.pptx)
references/           # artigo base, roteiro, TODO
```

## Arquitetura

**Contrato dos modelos (load-bearing).** Todo modelo em `models/` herda de
`conf.base_model.BaseModel` e expõe os mesmos três métodos, com um dict de retorno
uniforme:

- `grid_search(data)` — percorre o grid de hiperparâmetros e avalia cada candidato
  no split de **validação**, guardando o de menor erro (`update_best_model`).
- `get_predict(data)` — roda o `best_model` no split de **teste**.
- `train(data)` — `grid_search` + `get_predict`; devolve `lst_results`,
  `pred_test`, `pred_valid`, `best_rna`, `best_error`, `best_errors_list`.

Os notebooks e o ensemble consomem essas chaves pelo nome — **não** altere a
interface sem planejar.

**Processamento (`conf/processing.py`).** `DataProcessing(data, n_lags=5)`
transforma a série em janelas supervisionadas. Regras não-negociáveis:

- **Sem vazamento** — estatísticas Min-Max ajustadas **só no treino** e aplicadas a
  validação/teste.
- **Ordem temporal** — split cronológico 60/20/20, **sem shuffle**; nada de CV
  aleatório na série bruta.
- **Métricas em unidades originais** — reportadas sobre previsões *denormalizadas*
  (`denorm_data`), para comparar bases e modelos.

## Ambiente e execução

Python 3.12. Gerenciador: [`uv`](https://github.com/astral-sh/uv). Rodar tudo a
partir da raiz do repositório (para `conf/`, `models/`, `utils/` resolverem como
pacotes).

```bash
uv venv                                                   # cria .venv
uv pip install -r requirements.txt                        # deps de modelagem
uv pip install -r requirements.txt -r requirements-analysis.txt  # + análise/deck
```

`requirements-analysis.txt` (scikit-posthocs, ucimlrepo, python-pptx) é usado apenas
por análise/apresentação e **não** altera as versões fixadas de
scikit-learn/numpy/scipy.

```bash
uv run python benchmark.py        # tabela de métricas (todos os modelos × bases)
uv run jupyter lab                # abre os notebooks de experimento
```

> `scikit-learn >= 1.4` é obrigatório (`root_mean_squared_error`). Não altere as
> versões sem verificar — mudanças de default do sklearn podem alterar resultados.

## Resultados (6 bases)

Resposta honesta: **o GA é competitivo, mas não domina.**

- **Ranking médio (MSE):** MLP 2,83 (1º) · **GA 3,33 (2º)** · EMEAN 3,50 · ADA 3,50 ·
  ELM/SCN 4,67 · SVR 5,67 · **KNN 7,83 (pior)**.
- **Friedman** significativo (χ² = 18,67; p = 0,009), **mas** o pós-teste de
  **Nemenyi só separa o KNN** — os demais pares (inclusive GA×MLP e GA×EMEAN) são
  estatisticamente equivalentes. Com N = 6 bases e k = 8 métodos a distância crítica
  é larga → pós-teste com pouco poder.
- **GA vs. média (EMEAN)**, no mesmo pool top-4: vantagem pequena, vence em **3 de 6**
  bases → o ganho do aprendizado de pesos é marginal e inconsistente (o GA tende a
  superajustar a validação).

Resultados **indicativos, não conclusivos** (N pequeno — o próprio artigo base alerta
para isso). Artefatos (figuras/tabelas) são regeneráveis pelo
`global_metrics_forecast.ipynb` em `results/` (não versionado).

## Apresentação

O deck final (14 slides, PT-BR) está em `presentation/` — seções Introdução,
Método Proposto, Experimentos, Resultados (métricas, ranking, testes de hipótese,
dinâmica do GA) e Conclusão.
