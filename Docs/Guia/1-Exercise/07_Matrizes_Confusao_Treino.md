# Etapa 7 — Matrizes de confusão da avaliação no treino (fora da amostra)

Notebook: `Notebooks/Exercicio_01_Cancer_Recall.ipynb`, seção **4.2 Matriz de Confusão**.
Correspondência na referência: seção 4.2 de `Docs/reference/1-Arquivo.ipynb` (`cross_val_predict` + `confusion_matrix` + `ConfusionMatrixDisplay`, uma matriz por modelo). Foram acrescentadas as matrizes no limiar ajustado (F2) da Etapa 6 e um resumo das contagens.

---

## 1. Objetivo e relação com o exercício

As métricas das Etapas 5 e 6 resumem o desempenho em porcentagens. A matriz de confusão mostra **as contagens por trás delas**: quantos tumores de cada classe real foram classificados em cada classe prevista. Assim fica visível:

- quantos **cânceres não foram detectados** (FN), o erro mais grave do exercício;
- quantos **alarmes falsos** (FP) cada modelo gera;
- o efeito do limiar F2 em cada tipo de erro.

A especificação pede **matrizes separadas por modelo, com classes e eixos identificados**, e avaliação do treino **fora da amostra**. O teste continua reservado.

---

## 2. Código implementado, por blocos

### Bloco 1 — Explicação (Markdown)

Tabela TN/FP/FN/TP na posição em que aparecem nas matrizes e explicação de por que as previsões vêm de `cross_val_predict`.

### Bloco 2 — Função para desenhar a matriz

```python
from sklearn.metrics import ConfusionMatrixDisplay

def plot_matriz_confusao(y_real, y_pred, titulo):
    cm = confusion_matrix(y_real, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=['Benigno (0)', 'Maligno (1)'])
    display.plot(cmap=plt.cm.Blues)

    # Escreve a sigla de cada célula logo acima do valor (mesma cor do número, para manter o contraste)
    siglas = [['TN', 'FP'],
              ['FN', 'TP']]
    for i in range(2):          # i = linha (classe real)
        for j in range(2):      # j = coluna (classe prevista)
            valor = display.text_[i, j]
            valor.set_position((j, i + 0.12))
            display.ax_.text(j, i - 0.18, siglas[i][j], ha='center', va='center',
                             fontsize=13, fontweight='bold', color=valor.get_color())

    plt.title(titulo)
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.show()
    return cm
```

A base é o mesmo bloco da referência (`confusion_matrix` → `ConfusionMatrixDisplay` → `plot(cmap=Blues)`), com eixos e título explícitos. Virou função porque se repete **5 vezes aqui e 5 na Etapa 8**, o tipo de repetição relevante que a especificação permite eliminar. Ela também **retorna** a matriz, para montar o resumo.

**Siglas nas células (acrescentado a pedido, para facilitar a apresentação):**
- `siglas` reproduz a posição de cada caso na matriz: linha de cima (real Benigno) = TN e FP; linha de baixo (real Maligno) = FN e TP.
- `display.text_` é a matriz 2×2 dos textos com os números que o scikit-learn desenhou, e `display.ax_` é o gráfico.
- Em cada célula, o número é deslocado um pouco para baixo (`i + 0.12`) e a sigla é escrita um pouco acima (`i - 0.18`). No gráfico, o eixo Y cresce **para baixo**, então subtrair move o texto para cima.
- A sigla usa `valor.get_color()`, a cor que o scikit-learn escolheu para o número (branco em fundo escuro, azul-escuro em fundo claro), para manter o contraste.

### Bloco 3 — Dummy, SGD e Random Forest no limiar padrão

```python
y_train_pred_dummy = cross_val_predict(dummy_clf, X_train, y_train, cv=cv)
cm_treino_dummy = plot_matriz_confusao(y_train, y_train_pred_dummy, "Dummy Classifier - Treino (fora da amostra)")
```

Igual à referência, mas com o `cv` estratificado de 5 dobras. Sem `method`, o `cross_val_predict` usa `predict`, ou seja, o **limiar padrão** de cada modelo. O mesmo padrão se repete para `sgd_clf` e `forest_clf`.

### Bloco 4 — SGD e Random Forest no limiar F2

```python
y_train_pred_sgd_f2 = (y_scores_sgd >= limiar_sgd).astype(int)
cm_treino_sgd_f2 = plot_matriz_confusao(y_train, y_train_pred_sgd_f2, ...)
```

Reaproveita as pontuações fora da amostra da Etapa 6 (`y_scores_sgd`, `y_scores_forest`) e aplica o limiar escolhido. Não há treino novo. `.astype(int)` converte `True/False` em `1/0`.

### Bloco 5 — Resumo das contagens

```python
resumo_treino = pd.DataFrame(
    [cm.ravel() for cm in matrizes_treino.values()],
    index=matrizes_treino.keys(),
    columns=['TN', 'FP', 'FN', 'TP']
)
```

`cm.ravel()` transforma a matriz 2×2 `[[TN, FP], [FN, TP]]` na sequência `TN, FP, FN, TP`, que vira uma linha da tabela.

### Bloco 6 — Observações (Markdown)

Interpretação das matrizes no notebook.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `cross_val_predict(modelo, X_train, y_train, cv=cv)` | Uma previsão (classe) por tumor do treino, feita por um modelo que não o viu |
| `confusion_matrix(y_real, y_pred)` | Matriz 2×2: linhas = classe **real**, colunas = classe **prevista**, na ordem 0, 1 |
| `ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[...])` | Objeto de visualização; `display_labels` dá nome às classes |
| `display.plot(cmap=plt.cm.Blues)` | Desenha a matriz em tons de azul (mais escuro = mais tumores) |
| `plot_matriz_confusao(y_real, y_pred, titulo)` | Função do notebook: desenha a matriz com as siglas TN/FP/FN/TP e a retorna |
| `display.text_`, `display.ax_` | Atributos do `ConfusionMatrixDisplay` após o `plot`: textos dos valores (2×2) e eixo do gráfico |
| `ax.text(x, y, texto, ...)` | Escreve um texto na posição (coluna, linha) da matriz |
| `y_train_pred_*` | Previsões fora da amostra (padrão e F2) |
| `cm_treino_*` | Matrizes de confusão de cada modelo e limiar |
| `resumo_treino` | Tabela TN/FP/FN/TP das 5 matrizes |

---

## 4. Conceitos e fórmulas

**Leitura da matriz** (linhas = real, colunas = previsto):

$$\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$$

- A **diagonal principal** (TN, TP) contém os acertos; fora dela ficam os erros.
- A **linha de baixo** soma os malignos reais (170 no treino); a **linha de cima**, os benignos reais (285).

**Métricas a partir da matriz:**

$$Recall = \frac{TP}{TP + FN} \quad\text{(linha de baixo)} \qquad Precision = \frac{TP}{TP + FP} \quad\text{(coluna da direita)}$$

$$Especificidade = \frac{TN}{TN + FP} \quad\text{(linha de cima)} \qquad Acc = \frac{TN + TP}{\text{total}}$$

A **especificidade** é o "recall dos benignos": a fração de tumores benignos corretamente reconhecidos. Ela cai quando os FP aumentam.

**Por que "fora da amostra":** se o modelo treinado no treino inteiro classificasse o próprio treino, a matriz seria otimista, sobretudo na RF, que tende a memorizar os dados. Com `cross_val_predict`, cada tumor é classificado pelo modelo da dobra em que ele ficou de fora.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Uma figura por matriz | Exigência da especificação (matrizes separadas por modelo) e mesmo formato da referência |
| Eixos "Classe real" / "Classe prevista" e rótulos "Benigno (0)" / "Maligno (1)" | Exigência da especificação; os rótulos padrão do scikit-learn ("True label", "0", "1") são menos claros |
| Siglas TN/FP/FN/TP acima dos valores | Facilita a leitura pelo público na apresentação: cada número fica identificado sem precisar decorar a posição das células |
| Título com modelo, limiar e conjunto | Evita confundir matrizes do treino com as do teste (Etapa 8) e do limiar padrão com as do F2 |
| Mesmo `cv` das etapas anteriores | Os números ficam consistentes com as Etapas 5 e 6 (os FN e FP coincidem) |
| Matrizes F2 a partir de `y_scores_*` | Reaproveita as pontuações fora da amostra; um novo `cross_val_predict` daria os mesmos valores |
| Função `plot_matriz_confusao` | Elimina repetição relevante (10 matrizes no total) sem mudar o código da referência |
| Tabela-resumo | Permite comparar as 5 matrizes num único olhar, útil na apresentação |

---

## 6. Interpretação dos resultados obtidos

**Resumo das matrizes (treino, fora da amostra; 285 benignos e 170 malignos):**

| Modelo - limiar | TN | FP | FN | TP |
|---|---|---|---|---|
| Dummy | 285 | 0 | **170** | 0 |
| SGD - padrão | 278 | 7 | **7** | 163 |
| SGD - F2 | 272 | 13 | **4** | 166 |
| Random Forest - padrão | 275 | 10 | **9** | 161 |
| Random Forest - F2 | 272 | 13 | **6** | 164 |

**Métricas calculadas a partir dessas contagens:**

| Modelo - limiar | Recall (TP/170) | Precisão | Especificidade (TN/285) | Acurácia |
|---|---|---|---|---|
| Dummy | 0,00% | — (sem previsões positivas) | 100,00% | 62,64% |
| SGD - padrão | 95,88% | 95,88% (163/170) | 97,54% | 96,92% |
| SGD - F2 | 97,65% | 92,74% (166/179) | 95,44% | 96,26% |
| Random Forest - padrão | 94,71% | 94,15% (161/171) | 96,49% | 95,82% |
| Random Forest - F2 | 96,47% | 92,66% (164/177) | 95,44% | 95,82% |

**Leitura:**

1. **Dummy:** a matriz tem toda a coluna "Maligno" zerada. Os 170 malignos do treino seriam mandados para casa como benignos. A especificidade de 100% e a acurácia de 62,64% escondem esse fracasso total no recall.
2. **SGD:** no limiar padrão, os erros estão equilibrados (7 FN e 7 FP). Com o limiar F2, **3 malignos que eram perdidos passaram a ser detectados** (FN 7 → 4), e **6 benignos passaram a ser alarmes falsos** (FP 7 → 13).
3. **Random Forest:** no padrão, 9 FN e 10 FP. Com F2, FN 9 → 6 e FP 10 → 13 (3 a mais de cada lado).
4. **Comparação no F2:** os dois modelos chegam ao **mesmo número de FP (13)**, mas o SGD perde **2 malignos a menos** (4 contra 6). É uma diferença pequena, coerente com a Etapa 5, onde a diferença entre os modelos ficou dentro da variação entre dobras.
5. **Consistência:** os FN e FP do limiar padrão e do F2 são exatamente os da tabela da Etapa 6, e os recalls batem com a Etapa 5 (163/170 e 161/170).
6. **Acurácia × recall:** no SGD, mudar para o limiar F2 **reduziu a acurácia** (96,92% → 96,26%) e **aumentou o recall**. Para este problema, a acurácia sozinha indicaria a escolha errada.

**Cautela:** as matrizes "F2" usam as mesmas pontuações com que os limiares foram escolhidos. Por isso, tendem a ser um pouco **otimistas**. A verificação independente é o teste (Etapa 8).

---

## 7. Erros comuns e limitações

- **Inverter linhas e colunas:** no scikit-learn, **linhas = real** e **colunas = previsto**. Em outros materiais a convenção pode ser a oposta; os rótulos dos eixos evitam o engano.
- **Esquecer a ordem do `ravel()`:** é TN, FP, FN, TP (para classes 0 e 1).
- **Gerar a matriz com o modelo treinado no treino todo** (`forest_clf.predict(X_train)`): o resultado é otimista e enganoso.
- **Comparar matrizes com tamanhos de classe diferentes sem normalizar:** aqui todos usam o mesmo treino (285/170), então as contagens são comparáveis diretamente.
- **Limitação:** as matrizes F2 do treino não são uma avaliação independente do limiar (ver "Cautela" acima).
- **Limitação:** são 455 tumores; diferenças de 1 a 3 casos entre modelos não são conclusivas.

---

## 8. Perguntas que o professor poderia fazer

1. **Como ler a matriz?** Linhas são a classe real e colunas a classe prevista. A diagonal são os acertos; o canto inferior esquerdo é o FN (maligno previsto como benigno).
2. **Qual célula mais importa neste problema?** A dos FN, cânceres não detectados.
3. **Por que usar `cross_val_predict` e não o modelo já treinado?** Para que cada tumor seja classificado por um modelo que não o viu; senão a matriz seria otimista.
4. **O que mudou com o limiar F2 no SGD?** FN de 7 para 4 (3 cânceres a mais detectados) e FP de 7 para 13 (6 alarmes falsos a mais).
5. **O Dummy tem 100% de especificidade. Isso é bom?** Não. Ele chama tudo de benigno, então acerta todos os benignos e erra todos os 170 malignos.
6. **Por que as matrizes F2 podem ser otimistas?** Porque o limiar foi escolhido com essas mesmas previsões; o teste dará uma avaliação independente.
7. **Por que você criou uma função se a referência não criou?** O mesmo bloco se repete 10 vezes (5 no treino, 5 no teste); a função mantém o código da referência e evita repetição.

---

## 9. Como apresentar oralmente

> "A matriz de confusão mostra as contagens por trás das métricas: linhas são a classe real e colunas a prevista, e o canto inferior esquerdo são os falsos negativos, os cânceres não detectados. Gerei as previsões do treino fora da amostra com as mesmas 5 dobras, então cada tumor foi classificado por um modelo que não o tinha visto. O Dummy classifica tudo como benigno e perde os 170 malignos. O SGD, no limiar padrão, erra 7 malignos e 7 benignos. Com o limiar F2, passa a errar só 4 malignos, ao custo de 13 alarmes falsos. A Random Forest vai de 9 para 6 falsos negativos, com 13 falsos positivos. No limiar F2, os dois modelos têm os mesmos 13 falsos positivos, e o SGD perde dois malignos a menos. Como o limiar foi escolhido com esses mesmos dados, a confirmação vem no teste."

Mostrar na tela: a matriz do Dummy (coluna "Maligno" zerada), as duas matrizes do SGD lado a lado e a tabela-resumo.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Matriz de confusão**

| Item | Para conferir | Link |
|---|---|---|
| `confusion_matrix` | Convenção linhas = real, colunas = previsto; exemplo do `ravel()` com TN, FP, FN, TP | [scikit-learn — confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) |
| `ConfusionMatrixDisplay` | Parâmetros `display_labels`, `cmap`, o método `plot` e os atributos `text_` e `ax_` | [scikit-learn — ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) |
| Guia do usuário | Definição e leitura da matriz | [scikit-learn — Confusion matrix](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) |
| Exemplo oficial | Matrizes com e sem normalização | [scikit-learn — Confusion matrix example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) |
| API de visualização | Padrão `*Display` do scikit-learn | [scikit-learn — Visualizations](https://scikit-learn.org/stable/visualizations.html) |
| `cross_val_predict` | Previsões fora da amostra | [scikit-learn — cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) |

**NumPy, Matplotlib e Python**

| Item | Para conferir | Link |
|---|---|---|
| `ndarray.astype` | Converter `True/False` em `1/0` | [NumPy — ndarray.astype](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html) |
| `ravel` | "Achatar" a matriz 2×2 | [NumPy — numpy.ravel](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html) |
| `ax.text` | Escrever as siglas nas células (`ha`, `va`, `fontweight`, `color`) | [Matplotlib — Axes.text](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html) |
| `Text.set_position` | Deslocar o número para baixo da sigla | [Matplotlib — Text.set_position](https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text.set_position) |
| Mapas de cores | O `cmap` `Blues` | [Matplotlib — Choosing Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html) |
| Funções em Python | `def`, parâmetros e `return` | [Python — Defining Functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions) |
