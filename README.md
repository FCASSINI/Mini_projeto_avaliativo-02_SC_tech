# Credit Risk Prediction

Projeto acadêmico de Machine Learning desenvolvido na disciplina **Machine Learning e Visão Computacional**.

O projeto implementa um pipeline completo de classificação para estimar o risco de inadimplência de clientes, abrangendo análise exploratória, tratamento dos dados, feature engineering, balanceamento, modelagem, análise de overfitting e avaliação dos impactos de negócio.

---

## Problema de Negócio

Uma instituição financeira precisa avaliar o risco associado à concessão de crédito.

O objetivo do modelo é prever a variável `loan_status`:

- `0` — cliente não inadimplente;
- `1` — cliente inadimplente.

Uma classificação incorreta pode gerar impactos distintos:

- **Falso Positivo:** um bom pagador é classificado como inadimplente, podendo resultar em recusa indevida de crédito, perda de receita e insatisfação do cliente.
- **Falso Negativo:** um cliente inadimplente é classificado como seguro, podendo resultar em concessão de crédito com posterior perda financeira.

---

## Dataset

A base original contém:

- **32.581 registros**
- **12 variáveis**
- atributos numéricos e categóricos
- variável alvo: `loan_status`

Arquivo utilizado:

`credit_risk_dataset.csv`

### Principais variáveis

| Variável | Descrição |
|---|---|
| `person_age` | Idade do cliente |
| `person_income` | Renda do cliente |
| `person_home_ownership` | Situação de moradia |
| `person_emp_length` | Tempo de emprego |
| `loan_intent` | Finalidade do empréstimo |
| `loan_grade` | Classificação de risco do empréstimo |
| `loan_amnt` | Valor do empréstimo |
| `loan_int_rate` | Taxa de juros |
| `loan_status` | Situação de inadimplência |
| `loan_percent_income` | Proporção da renda comprometida |
| `cb_person_default_on_file` | Histórico de inadimplência |
| `cb_person_cred_hist_length` | Tempo de histórico de crédito |
| `comprometimento_renda` | Percentual da renda comprometido com o empréstimo |

---

## Metodologia

O projeto foi desenvolvido em seis etapas principais:

1. Análise Exploratória de Dados (EDA)
2. Tratamento e limpeza
3. Feature Engineering
4. Preparação para Machine Learning
5. Modelagem e diagnóstico de overfitting
6. Avaliação dos modelos e análise de negócio

---

## 1. Análise Exploratória de Dados

A análise inicial revelou:

- desbalanceamento da variável alvo;
- presença de valores ausentes;
- registros duplicados;
- distribuições assimétricas em algumas variáveis;
- presença de valores estatisticamente extremos;
- associação relevante entre algumas variáveis e inadimplência.

Foram identificados **165 registros duplicados**.

Os principais valores ausentes estavam em:

- `person_emp_length`;
- `loan_int_rate`.

A análise pelo método IQR também revelou valores extremos em variáveis como idade, renda, tempo de emprego e valor do empréstimo.

Entretanto, os outliers não foram removidos automaticamente, pois muitos representavam situações plausíveis e sua exclusão poderia eliminar informações legítimas.

### Grade de crédito e inadimplência

A proporção de inadimplência aumentou de maneira expressiva nas grades de maior risco:

| Grade | Inadimplência |
|---|---:|
| A | 9,96% |
| B | 16,28% |
| C | 20,73% |
| D | 59,05% |
| E | 64,42% |
| F | 70,54% |
| G | 98,44% |

A matriz de correlação também mostrou associação positiva entre `loan_status` e:

- `loan_percent_income`: aproximadamente **0,38**;
- `loan_int_rate`: aproximadamente **0,34**.

---

## 2. Tratamento dos Dados

Os **165 registros duplicados** foram removidos.

A base passou de:

- 32.581 registros
- para **32.416 registros**

Após a remoção de duplicados, foram identificados:

- `person_emp_length`: **887 valores ausentes (2,74%)**
- `loan_int_rate`: **3.095 valores ausentes (9,55%)**

Os valores ausentes foram tratados utilizando a mediana:

- `person_emp_length`: **4,0**
- `loan_int_rate`: **10,99**

A mediana foi escolhida por sua maior robustez diante de distribuições assimétricas e valores extremos.

### Tratamento de outliers

Os outliers foram identificados e quantificados pelo método do intervalo interquartil (IQR).

Como muitos valores classificados estatisticamente como extremos ainda eram plausíveis no contexto do problema, optou-se por **manter os outliers**, evitando remoções ou clipping indiscriminados.

Essa decisão foi considerada especialmente relevante para o KNN, que é mais sensível a distâncias e valores extremos, enquanto a Árvore de Decisão tende a ser mais robusta a esse tipo de característica.

---

## 3. Feature Engineering

Foi criada a variável:

```python
comprometimento_renda = (loan_amnt / person_income) * 100
```

A nova feature apresentou:

- média: **17,06%**
- mediana: **14,81%**
- máximo: **83%**

A correlação entre `comprometimento_renda` e `loan_percent_income` foi de aproximadamente **0,999**, indicando forte redundância.

Por esse motivo, `loan_percent_income` foi removida do conjunto de preditores, mantendo-se `comprometimento_renda`.

---

## 4. Preparação para Modelagem

As variáveis categóricas foram convertidas utilizando **One-Hot Encoding com `pandas.get_dummies()`**.

Os dados foram divididos em:

- **80% treino**
- **20% teste**

utilizando:

```python
stratify=y
random_state=42
```

para preservar a proporção das classes e garantir reprodutibilidade.

### Balanceamento

Foi aplicado **Random Under Sampling** exclusivamente ao conjunto de treino.

O conjunto de teste permaneceu com sua distribuição original para evitar vazamento de dados e permitir uma avaliação mais próxima da realidade.

A escolha pelo undersampling foi motivada pela quantidade suficiente de registros disponíveis, pela simplicidade metodológica e pela ausência de criação de observações sintéticas.

### Escalonamento

O `StandardScaler` foi aplicado somente às variáveis numéricas utilizadas pelo modelo KNN.

O escalonador foi ajustado nos dados de treinamento balanceados e, posteriormente, aplicado ao conjunto de teste apenas com `transform`, evitando vazamento de informações.

A Árvore de Decisão utilizou os dados sem escalonamento.

---

## 5. Modelagem e Overfitting

Foram avaliados dois algoritmos:

- K-Nearest Neighbors (KNN)
- Decision Tree

### KNN

Foram testados:

`K = 3, 5, 7 e 9`

| K | Acurácia Treino | Acurácia Teste | Diferença |
|---:|---:|---:|---:|
| 3 | 89,06% | 80,57% | 8,49 p.p. |
| 5 | 86,11% | 82,14% | 3,97 p.p. |
| 7 | 85,14% | 81,89% | 3,25 p.p. |
| 9 | 84,37% | 82,77% | 1,59 p.p. |

A configuração selecionada foi:

**KNN com K = 9**

por apresentar o melhor equilíbrio entre desempenho no teste e diferença entre treino e teste.

### Árvore de Decisão

Foram avaliadas diferentes profundidades:

| max_depth | Acurácia Treino | Acurácia Teste | Diferença |
|---|---:|---:|---:|
| 2 | 77,87% | 82,37% | -4,50 p.p. |
| 5 | 83,50% | 88,23% | -4,73 p.p. |
| 7 | 85,81% | 90,25% | -4,44 p.p. |
| None | 100,00% | 80,35% | 19,65 p.p. |

A árvore sem limite de profundidade apresentou forte sinal de overfitting: atingiu 100% no treinamento, mas caiu para aproximadamente 80% no teste.

A configuração selecionada foi:

**Decision Tree com `max_depth=7`**

---

## 6. Avaliação Final

### KNN — K = 9

| Métrica da classe inadimplente | Resultado |
|---|---:|
| Precision | 57,94% |
| Recall | 77,43% |
| F1-score | 66,28% |
| Accuracy geral | 82,77% |

Matriz de confusão:

- Verdadeiros Negativos: **4.269**
- Falsos Positivos: **797**
- Falsos Negativos: **320**
- Verdadeiros Positivos: **1.098**

### Árvore de Decisão — max_depth = 7

| Métrica da classe inadimplente | Resultado |
|---|---:|
| Precision | 80,56% |
| Recall | 73,06% |
| F1-score | 76,63% |
| Accuracy geral | 90,25% |

Matriz de confusão:

- Verdadeiros Negativos: **4.816**
- Falsos Positivos: **250**
- Falsos Negativos: **382**
- Verdadeiros Positivos: **1.036**

---

## Veredito de Negócio

O falso negativo representa um risco particularmente relevante para uma instituição financeira, pois corresponde a um cliente inadimplente classificado como seguro.

O KNN apresentou maior recall para a classe inadimplente, identificando **77,43%** desses clientes, contra **73,06%** da Árvore de Decisão.

Entretanto, essa vantagem correspondeu a **62 inadimplentes adicionais identificados**, enquanto o KNN produziu **547 falsos positivos adicionais**.

A Árvore de Decisão apresentou:

- maior acurácia;
- maior precision para inadimplentes;
- maior F1-score para inadimplentes;
- redução expressiva de falsos positivos;
- bom recall da classe de risco;
- melhor equilíbrio global entre os tipos de erro.

Por esse motivo, **a Árvore de Decisão com `max_depth=7` foi selecionada como a melhor candidata entre os modelos avaliados neste experimento**.

---

## Limitações

Este projeto possui finalidade acadêmica e algumas limitações devem ser consideradas antes de qualquer aplicação real:

- o dataset não fornece o custo monetário real de falsos positivos e falsos negativos;
- o balanceamento por undersampling remove parte das observações da classe majoritária;
- foram avaliados apenas KNN e Árvore de Decisão;
- os resultados foram obtidos a partir de uma única divisão treino/teste;
- uma aplicação real exigiria validação adicional, novos dados e análise dos custos financeiros envolvidos nos erros do modelo;
- o modelo não deve ser utilizado isoladamente como mecanismo automático de decisão de crédito sem validações adicionais.

---

## Estrutura do Repositório

```text
.
├── data/
│   └── credit_risk_dataset.csv
├── notebooks/
│   └── 01_eda_credit_risk.ipynb
├── README.md
└── requirements.txt
```

---

## Tecnologias

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

---

## Status

**Projeto concluído.**

Melhor configuração avaliada:

```python
DecisionTreeClassifier(
    max_depth=7,
    random_state=42
)
```

Accuracy no conjunto de teste: **90,25%**.
