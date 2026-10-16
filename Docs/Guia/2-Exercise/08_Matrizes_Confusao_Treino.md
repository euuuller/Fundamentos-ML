# Etapa 8 — Matrizes de confusão da avaliação no treino (fora da amostra)

Notebook: `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`, seção **4.2 Matriz de Confusão**.
Correspondência na referência: seção 4.2 de `Docs/reference/1-Arquivo.ipynb` (`cross_val_predict` + `ConfusionMatrixDisplay`, uma matriz por modelo). Mesma estrutura da Etapa 7 do Exercício 01, incluindo a função com siglas TN/FP/FN/TP pedida para a apresentação.

---

## 1. Objetivo e relação com o exercício

O enunciado inclui a **matriz de confusão** entre as medidas de desempenho. Esta etapa mostra as **contagens** por trás das métricas das Etapas 6 e 7:

- quantos sobreviventes cada modelo **deixou de identificar** (FN);
- quantos não sobreviventes ele **apontou como sobreviventes** (FP);
- como o limiar F1 redistribuiu esses erros.

São **5 matrizes separadas** (exigência da especificação), todas do treino com previsões fora da amostra. O teste continua reservado.

---

## 2. Código implementado, por blocos

### Bloco 1 — Explicação (Markdown)

Tabela TN/FP/FN/TP na posição em que aparecem na matriz, com as classes do Titanic.

### Bloco 2 — Função para desenhar a matriz

```python
def plot_matriz_confusao(y_real, y_pred, titulo):
    cm = confusion_matrix(y_real, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=['Não Sobreviveu (0)', 'Sobreviveu (1)'])
    display.plot(cmap=plt.cm.Blues)
    siglas = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            valor = display.text_[i, j]
            valor.set_position((j, i + 0.12))
            display.ax_.text(j, i - 0.18, siglas[i][j], ..., color=valor.get_color())
    plt.title(titulo); plt.xlabel("Classe prevista"); plt.ylabel("Classe real")
    plt.show()
    return cm
```

É a mesma função do Exercício 01 (com as siglas acima de cada valor), só com os **rótulos do Titanic**. Será reutilizada nas 5 matrizes do teste (Etapa 9).

### Bloco 3 — Limiar padrão (Dummy, SGD, RF)

```python
y_train_pred_dummy = cross_val_predict(dummy_clf, X_train, y_train, cv=cv)
cm_treino_dummy = plot_matriz_confusao(y_train, y_train_pred_dummy, "Dummy Classifier - Treino (fora da amostra)")
```

Sem `method`, o `cross_val_predict` usa `predict`, ou seja, o limiar padrão. Repetido para `sgd_clf` e `forest_clf`.

### Bloco 4 — Limiar F1 (SGD, RF)

```python
y_train_pred_sgd_f1 = (y_scores_sgd >= limiar_sgd).astype(int)
```

Aplica os limiares da Etapa 7 às pontuações fora da amostra já calculadas, sem novo treino.

### Bloco 5 — Resumo

```python
resumo_treino = pd.DataFrame([cm.ravel() for cm in matrizes_treino.values()], ...,
                             columns=['TN', 'FP', 'FN', 'TP'])
resumo_treino['Erros (FN + FP)'] = resumo_treino['FN'] + resumo_treino['FP']
```

Em relação ao Exercício 01, foi acrescentada a coluna **total de erros**: como aqui FN e FP têm peso igual, o total é uma leitura natural.

### Bloco 6 — Observações (Markdown)

Interpretação no notebook.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `cross_val_predict(modelo, X_train, y_train, cv=cv)` | Uma previsão por passageiro, feita por um modelo que não o viu |
| `confusion_matrix(y_real, y_pred)` | Matriz 2×2: linhas = real, colunas = previsto |
| `ConfusionMatrixDisplay(..., display_labels=[...])` | Desenho com os nomes das classes |
| `display.text_`, `display.ax_` | Textos dos valores e eixo do gráfico (para as siglas) |
| `plot_matriz_confusao` | Desenha com siglas e retorna a matriz |
| `y_train_pred_*` | Previsões fora da amostra (padrão e F1) |
| `cm_treino_*` | As 5 matrizes |
| `resumo_treino` | TN, FP, FN, TP e total de erros |

---

## 4. Conceitos e fórmulas

$$\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$$

- Linha de cima = **não sobreviventes reais** (439 no treino); linha de baixo = **sobreviventes reais** (273).
- **FN:** sobrevivente previsto como não sobrevivente; **FP:** o inverso.

**Métricas a partir da matriz:**

$$Recall = \frac{TP}{273} \qquad Especificidade = \frac{TN}{439} \qquad Precision = \frac{TP}{TP+FP} \qquad Acc = \frac{TN+TP}{712}$$

**Total de erros e F1:** com FN e FP de mesmo peso, reduzir o total de erros é desejável, mas o F1 não é igual à acurácia: ele ignora os TN e olha só a classe positiva. Por isso, um ajuste pode manter o total de erros e ainda assim melhorar o F1, redistribuindo os erros, como aconteceu no SGD.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Uma figura por matriz, com siglas | Exigência da especificação e pedido do usuário (apresentação) |
| Mesma função do Exercício 01 | Consistência entre os exercícios; só os rótulos mudam |
| Mesmo `cv` das Etapas 6 e 7 | Números consistentes entre as etapas |
| Matrizes F1 a partir de `y_scores_*` | Reaproveita as pontuações fora da amostra |
| Coluna "Erros (FN + FP)" | Leitura natural quando os dois erros têm o mesmo peso |

---

## 6. Interpretação dos resultados obtidos

**Resumo (treino, fora da amostra; 439 não sobreviventes e 273 sobreviventes):**

| Modelo - limiar | TN | FP | FN | TP | Erros (FN + FP) |
|---|---|---|---|---|---|
| Dummy | 439 | 0 | **273** | 0 | 273 |
| SGD - padrão | 322 | 117 | 83 | 190 | 200 |
| SGD - F1 | 301 | 138 | 61 | 212 | 199 |
| Random Forest - padrão | 355 | 84 | 69 | 204 | 153 |
| Random Forest - F1 | 378 | 61 | 76 | 197 | **137** |

**Métricas calculadas a partir das contagens:**

| Modelo - limiar | Recall (TP/273) | Especificidade (TN/439) | Precisão | Acurácia |
|---|---|---|---|---|
| Dummy | 0,00% | 100,00% | — | 61,66% |
| SGD - padrão | 69,60% | 73,35% | 61,89% (190/307) | 71,91% |
| SGD - F1 | 77,66% | 68,56% | 60,57% (212/350) | 72,05% |
| RF - padrão | 74,73% | 80,87% | 70,83% (204/288) | 78,51% |
| RF - F1 | 72,16% | **86,10%** | **76,36%** (197/258) | **80,76%** |

**Leitura:**

1. **Dummy:** a coluna "Sobreviveu" está zerada, então os 273 sobreviventes viram FN. A especificidade de 100% e a acurácia de 61,66% escondem que ele não identifica ninguém.
2. **SGD:** o limiar F1 trocou **22 FN por 21 FP**. O total de erros quase não mudou (200 → 199), mas os erros ficaram mais equilibrados, e o F1 subiu (Etapa 7).
3. **Random Forest:** o limiar F1 trocou **23 FP por 7 FN**. O total de erros caiu de 153 para **137**, o menor de todas as matrizes, e a acurácia subiu para 80,76%.
4. **RF × SGD:** a RF erra menos nos dois limiares (153 e 137 contra 200 e 199), coerente com a AUC maior (0,862 × 0,787).
5. **Consistência:** os FN e FP coincidem exatamente com a tabela da Etapa 7, e os recalls batem com a Etapa 6.
6. **Cautela:** as matrizes F1 usam as mesmas pontuações com que os limiares foram escolhidos, então tendem a ser um pouco otimistas. A confirmação vem do teste (Etapa 9).

---

## 7. Erros comuns e limitações

- **Inverter linhas e colunas:** no scikit-learn, linhas = real e colunas = previsto. Os rótulos dos eixos evitam o engano.
- **Gerar a matriz com o modelo treinado no treino inteiro:** a matriz seria otimista, sobretudo para a RF, que memoriza os dados.
- **Achar que menos erros totais sempre significa F1 maior:** o SGD manteve o total e melhorou o F1 só redistribuindo os erros.
- **Limitação:** as matrizes F1 do treino não são uma avaliação independente do limiar.

---

## 8. Perguntas que o professor poderia fazer

1. **O que significa FN aqui?** Um sobrevivente que o modelo previu como não sobrevivente.
2. **Qual configuração erra menos no treino?** A Random Forest com limiar F1: 137 erros (61 FP e 76 FN).
3. **O ajuste do SGD reduziu os erros?** Praticamente não (200 → 199), mas os equilibrou: 22 FN a menos e 21 FP a mais, e isso melhorou o F1.
4. **Por que o Dummy tem especificidade de 100%?** Ele prevê "não sobreviveu" para todos: acerta todos os não sobreviventes e erra todos os sobreviventes.
5. **Por que usar previsões fora da amostra?** Para que cada passageiro seja classificado por um modelo que não o viu; senão a matriz seria otimista.
6. **Essas matrizes F1 são confiáveis?** São um pouco otimistas, porque o limiar foi escolhido com os mesmos dados; o teste confirma.

---

## 9. Como apresentar oralmente

> "As matrizes mostram as contagens por trás das métricas: linhas são a classe real e colunas a prevista, e cada célula tem sua sigla. O Dummy classifica todos como não sobreviventes, e os 273 sobreviventes do treino viram falsos negativos. No SGD, o limiar F1 trocou 22 falsos negativos por 21 falsos positivos: o total de erros ficou igual, mas mais equilibrado. Na Random Forest, o limiar F1 cortou 23 falsos positivos e acrescentou 7 falsos negativos, e o total de erros caiu de 153 para 137, o menor de todos. A Random Forest erra menos que o SGD nos dois limiares. Como os limiares vieram desses mesmos dados, a confirmação vem no teste."

Mostrar na tela: a matriz do Dummy, as duas da RF e a tabela-resumo com o total de erros.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Matriz de confusão**

| Item | Para conferir | Link |
|---|---|---|
| `confusion_matrix` | Convenção linhas/colunas e `ravel()` | [scikit-learn — confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) |
| `ConfusionMatrixDisplay` | `display_labels`, `plot`, `text_`, `ax_` | [scikit-learn — ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) |
| Guia do usuário | Leitura da matriz | [scikit-learn — Confusion matrix](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) |
| `cross_val_predict` | Previsões fora da amostra | [scikit-learn — cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) |

**Matplotlib e NumPy**

| Item | Para conferir | Link |
|---|---|---|
| `ax.text` | Siglas nas células | [Matplotlib — Axes.text](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html) |
| `Text.set_position` | Deslocar o número | [Matplotlib — Text.set_position](https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text.set_position) |
| `ravel` | Matriz 2×2 → TN, FP, FN, TP | [NumPy — numpy.ravel](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html) |
| `astype` | `True/False` → `1/0` | [NumPy — ndarray.astype](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html) |
