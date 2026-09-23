# Credit Risk Prediction

Projeto acadêmico de Machine Learning desenvolvido como parte da disciplina
**Machine Learning e Visão Computacional**.

## Contexto

Instituições financeiras precisam avaliar o risco associado à concessão de crédito.
Uma decisão incorreta pode gerar impactos tanto financeiros quanto comerciais.

Neste projeto será desenvolvido um pipeline de Machine Learning para prever
a inadimplência de clientes a partir de características relacionadas ao perfil
do cliente e ao empréstimo solicitado.

## Objetivo

Construir e avaliar modelos de classificação capazes de prever a variável
`loan_status`:

- `0` — cliente não inadimplente;
- `1` — cliente inadimplente.

O projeto comparará dois algoritmos:

- K-Nearest Neighbors (KNN);
- Decision Tree.

Além do desempenho preditivo, será analisado o impacto dos falsos positivos
e falsos negativos sob a perspectiva do negócio.

## Dataset

Arquivo utilizado:

`credit_risk_dataset.csv`

A base contém inicialmente:

- 32.581 registros;
- 12 variáveis;
- variáveis numéricas e categóricas;
- variável alvo: `loan_status`.

A análise detalhada da qualidade e distribuição dos dados será realizada
durante a etapa de Análise Exploratória de Dados (EDA).

## Etapas do projeto

1. Estruturação do projeto
2. Análise Exploratória de Dados (EDA)
3. Tratamento e limpeza dos dados
4. Feature Engineering
5. Preparação dos dados para Machine Learning
6. Modelagem com KNN e Árvore de Decisão
7. Análise de overfitting e seleção de hiperparâmetros
8. Avaliação dos modelos
9. Análise de impacto de negócio

## Estrutura do repositório

```text
.
├── data/
│   └── credit_risk_dataset.csv
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore