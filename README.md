# Análise Preditiva de Preços de Combustíveis

Dashboard interativo para análise e previsão dos preços de combustíveis no
Brasil, com base em dados históricos da ANP e variáveis macroeconômicas
(câmbio USD/BRL, IPCA, petróleo Brent). Projeto prático do TCC
`ANÁLISE PREDITIVA DE PREÇOS DE COMBUSTÍVEIS - Dashboard com Simulação de
Cenários Macroeconômicos.pdf` (Fatec Santana de Parnaíba).

## Setup do ambiente

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e preencha as credenciais (nenhum dado
sensível deve ser commitado — `.env` já está no `.gitignore`):

```bash
copy .env.example .env
```

### 1. Google Cloud + BigQuery (para os dados da ANP via Base dos Dados)

A série histórica de preços da ANP é consultada via
[Base dos Dados](https://basedosdados.org/dataset/6ea3e28a-42be-401a-a066-ad87ca931e69)
(tabela `basedosdados.br_anp_precos_combustiveis.microdados`), que exige um
projeto Google Cloud com a API do BigQuery ativada:

1. Entre em [console.cloud.google.com](https://console.cloud.google.com/)
   com uma conta Google (a mesma do Gmail serve) e aceite os termos, se for
   a primeira vez.
2. Crie um novo projeto (canto superior esquerdo → "Novo projeto") e anote o
   **Project ID** (não é o nome, é o ID — geralmente com um sufixo
   numérico).
3. No menu, vá em "APIs e serviços" → "Biblioteca" e ative a **BigQuery
   API** para esse projeto.
4. Preencha `BASEDOSDADOS_PROJECT_ID` no `.env` com o Project ID do passo 2.

Não é preciso instalar o Google Cloud CLI: o pacote `basedosdados` usa
`pydata-google-auth`, que na primeira consulta abre uma aba no navegador
pedindo para você fazer login com a conta Google do projeto e autorizar o
acesso ao BigQuery. Depois disso o token fica em cache localmente
(`~/.config/pydata_google_auth/`) e as próximas execuções não pedem login de
novo. O BigQuery tem um nível gratuito de 1 TB de consulta por mês — mais
que suficiente para este projeto — e não exige cartão de crédito para isso.

Cobertura gratuita da tabela: 2004-05-10 até 2026-08-08 (a semana mais
recente exige o plano pago BD Pro). `src/data/collect_anp.py` já usa esse
corte por padrão.

### 2. API key da EIA (para o preço do Brent)

1. Cadastro gratuito (só e-mail) em
   [eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php).
2. Preencha `EIA_API_KEY` no `.env` com a chave recebida por e-mail.

### 3. Câmbio USD/BRL e IPCA

Não precisam de cadastro — são coletados diretamente da API pública do
Sistema Gerenciador de Séries Temporais (SGS) do Banco Central do Brasil.

## Coletando os dados

```bash
python -m src.data.collect_bcb        # câmbio + IPCA (não precisa de credenciais)
python -m src.data.collect_anp        # preços da ANP (precisa do setup do item 1)
python -m src.data.collect_eia_brent  # Brent (precisa do setup do item 2)
```

Cada script salva o resultado em `data/raw/` (parquet) e não baixa de novo
se o arquivo já existir — use `--force` para refazer. `collect_anp.py` e
`collect_eia_brent.py` aceitam `--probe-schema` / `--probe` para checar,
com uma consulta barata de metadados, se os nomes de coluna/facet usados no
código batem com os da fonte.

## Limpeza, compatibilização temporal e banco de dados

```bash
python -m src.processing.clean_anp        # agrega a ANP para granularidade semanal
python -m src.processing.clean_macro      # câmbio/Brent (diário -> semanal) + IPCA (mensal -> semanal, forward fill)
python -m src.processing.build_database   # monta data/fuel_prices.db a partir dos dois anteriores
```

Saída em `data/processed/` (parquet) e `data/fuel_prices.db` (SQLite, schema
relacional: `dim_municipio`, `precos_semanais`, `macro_semanal`, com índices
por município e por produto). Todos aceitam `--force` para refazer.

## Modelo preditivo (Random Forest)

```bash
python -m src.models.train_random_forest --municipio 3550308 --produto Gasolina
```

Treina e avalia um modelo por (município, produto): validação cruzada com
divisão temporal (`TimeSeriesSplit`) + holdout final (últimas 52 semanas,
nunca usadas em treino). Salva o modelo "implantado" (treinado em 100% dos
dados) e as métricas/importância de features em
`models/rf_<id_municipio>_<produto>.joblib` / `.json` (pasta não
versionada). `--municipio` é o código IBGE (consulte `dim_municipio` no
banco); `--n-splits` e `--semanas-holdout` são opcionais.

## Dashboard

```bash
streamlit run src/app/dashboard.py
```

Filtros por estado/município/combustível na barra lateral, KPIs, gráfico
histórico (Plotly, com faixa mín-máx e semanas marcadas como outlier),
tabela comparativa entre combustíveis na cidade escolhida e, quando já
existe modelo treinado para a combinação (ver seção anterior), uma
previsão simples para a semana seguinte — com botão para treinar na hora
se ainda não existir.

Quando há modelo treinado, aparece também uma seção de
**interpretabilidade (SHAP)**: importância global das features (gráfico
de barras, agrupado em preço/histórico, macro e geopolítico — responde
diretamente câmbio vs. Brent, qual pesa mais) e uma explicação local em
cascata (waterfall) de por que o modelo previu aquele valor específico
para a semana seguinte.

Por fim, uma **simulação de cenários (what-if)**: ajuste câmbio, Brent e
IPCA e veja o impacto estimado em tempo real, mais um toggle para simular
um novo choque geopolítico (reaproveita a feature da invasão da Ucrânia,
único evento geopolítico codificado no modelo). Valores fora da faixa
vista no treino disparam um aviso sobre a limitação de extrapolação do
Random Forest — o valor não é bloqueado, só sinalizado.

## Testes

```bash
pytest
```

## Estrutura

```
src/
  config.py          caminhos do projeto e leitura de credenciais (.env)
  data/
    collect_bcb.py        câmbio USD/BRL e IPCA (Banco Central, SGS)
    collect_anp.py        preços de combustíveis (ANP via Base dos Dados/BigQuery)
    collect_eia_brent.py  preço do petróleo Brent (EIA)
  processing/
    clean_anp.py           limpeza + agregação semanal dos preços da ANP
    clean_macro.py          compatibilização temporal de câmbio/Brent/IPCA
    build_database.py       monta o SQLite relacional (dim_municipio, precos_semanais, macro_semanal)
  features/
    geopolitical_events.py  codifica eventos geopolíticos (dummy + semanas desde o início)
    build_features.py       lags, janelas móveis, join com macro, alvo t+1
  models/
    train_random_forest.py  treino + validação temporal + holdout + persistência do modelo
    explain.py               interpretabilidade SHAP (importância global + explicação local)
  app/
    data_access.py         acesso cacheado ao SQLite para o dashboard
    dashboard.py            página Streamlit (filtros, gráficos, previsão, SHAP, what-if)
data/
  raw/                dados brutos coletados (não versionado)
  processed/          dados tratados, granularidade semanal (não versionado)
  fuel_prices.db      banco SQLite relacional (não versionado)
models/                modelos treinados (.joblib) + métricas (.json), não versionado
notebooks/            exploração
tests/                testes automatizados
```
