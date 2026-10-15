# Etapa 9 — Comparação final, interpretação dos resultados e conclusão

Notebook: `Notebooks/Exercicio_01_Cancer_Recall.ipynb`, seção **6. Comparação Final e Conclusão** (tabela e gráfico comparativos, 6.1 Interpretação, 6.2 Conclusão).
Correspondência na referência: não há seção equivalente; a referência termina no resumo do teste (5.4). Esta etapa fecha o exercício, como prevê a especificação.

---

## 1. Objetivo e relação com o exercício

Juntar tudo o que foi medido e responder às perguntas do exercício:

- Os três modelos funcionam? Como se comparam em **acurácia, precisão, recall e F1**?
- O **recall** (detecção de malignos) pôde ser melhorado? A que custo?
- O que se espera em dados novos (teste) em relação ao que foi estimado no treino?

Além disso, esta etapa **verifica a reprodutibilidade**: o notebook inteiro foi reexecutado do zero, em kernel limpo.

---

## 2. Código implementado, por blocos

### Bloco 1 — Tabela final treino × teste

```python
previsoes_finais = {
    'Dummy': (y_train_pred_dummy, y_pred_dummy),
    'SGD - padrão': (y_train_pred_sgd, y_pred_sgd),
    ...
}
for modelo, (pred_treino, pred_teste) in previsoes_finais.items():
    for conjunto, y_real, y_pred in [('Treino (validação cruzada)', y_train, pred_treino),
                                     ('Teste', y_test, pred_teste)]:
        linhas.append({... 'Recall': recall_score(y_real, y_pred), ...,
                       'FN': ((y_real == 1) & (y_pred == 0)).sum(),
                       'FP': ((y_real == 0) & (y_pred == 1)).sum(), ...})
df_final = pd.DataFrame(linhas).set_index(['Modelo', 'Conjunto'])
```

- Reaproveita as previsões já calculadas: as do treino, fora da amostra, da Etapa 7, e as do teste, da Etapa 8. **Nada é treinado nem ajustado de novo.**
- FN e FP são contados com máscaras booleanas: "real = 1 **e** previsto = 0" é um falso negativo.
- `set_index(['Modelo', 'Conjunto'])` cria um índice de dois níveis, com cada modelo em duas linhas (treino e teste).

### Bloco 2 — Gráfico treino × teste (recall e precisão)

```python
tabela = df_final[metrica].unstack('Conjunto').loc[modelos_reais, ['Treino (validação cruzada)', 'Teste']]
tabela.plot(kind='bar', ...)
ax.set_ylim(0.8, 1.07)
ax.set_yticks(np.arange(0.80, 1.001, 0.05))
ax.set_ylabel("Eixo começa em 80% (diferenças ampliadas)")
```

- `unstack('Conjunto')` transforma o nível "Conjunto" do índice em colunas (Treino | Teste), o formato que o gráfico de barras agrupadas precisa.
- O Dummy ficou fora do gráfico porque seus valores (0%) já estão na tabela e achatariam a escala.
- **O eixo começa em 80%** para que as diferenças de poucos pontos fiquem visíveis. Como isso amplia visualmente as diferenças, o próprio eixo avisa.

### Bloco 3 — Interpretação e conclusão (Markdown)

**6.1 Interpretação** (seis pontos numerados) e **6.2 Conclusão**, com limitações e o aviso de que não há validade clínica. Todos os números citados vêm da tabela final.

### Bloco 4 — Verificação de reprodutibilidade (fora do notebook)

1. As saídas de texto de todas as células de código foram salvas.
2. O notebook foi reexecutado do início ao fim em **kernel limpo** (`jupyter nbconvert --execute --inplace`).
3. As saídas novas foram comparadas célula a célula com as anteriores.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `previsoes_finais` | Para cada variante: (previsões do treino fora da amostra, previsões do teste) |
| `df_final` | Tabela final com índice (Modelo, Conjunto): recall, precisão, F1, acurácia, FN, FP, nº de malignos |
| `DataFrame.set_index([...])` | Índice de dois níveis (*MultiIndex*) |
| `Series.unstack('Conjunto')` | Passa um nível do índice para as colunas |
| `ax.set_yticks(np.arange(...))` | Define as marcas do eixo Y (até 100%) |
| `modelos_reais` | As 4 variantes de SGD/RF exibidas no gráfico |

---

## 4. Conceitos e fórmulas

**Comparação treino (fora da amostra) × teste:** as duas estimativas usam previsões de modelos que não viram os tumores avaliados, mas diferem em dois pontos:

1. **Tamanho:** 455 tumores (170 malignos) no treino e 114 (42 malignos) no teste. A incerteza do teste é maior: um maligno vale $1/42 \approx 2{,}38$ p.p. de recall, contra $1/170 \approx 0{,}59$ p.p. no treino.
2. **Uso nas decisões:** os limiares F2 foram escolhidos com as previsões do treino (levemente otimistas para o F2); o teste não participou de nenhuma decisão.

**Diferença em número de tumores:**

$$\Delta FN \approx \Delta Recall \times n_{malignos} \qquad \text{(ex.: } 2{,}4\ \text{p.p.} \times 42 \approx 1 \text{ tumor)}$$

Traduzir diferenças percentuais em **número de tumores** ajuda a não superestimar diferenças pequenas.

**Reprodutibilidade:** com `random_state=42` em todos os pontos aleatórios (divisão, dobras, SGD, RF), a mesma execução produz exatamente os mesmos números.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Treino (fora da amostra) e teste lado a lado | Mostra se o comportamento estimado se confirmou em dados novos; a especificação pede comparação final |
| Previsões agrupadas (*pooled*) no treino | Permitem comparar contagens (FN/FP) diretamente com o teste. Os recalls agrupados coincidem com a Etapa 5, e as precisões diferem minimamente das médias por dobra |
| Contagens (FN, FP, Malignos) junto das porcentagens | Evitam interpretar diferenças de 1 tumor como "grandes" |
| Eixo a partir de 80% com aviso | Equilíbrio entre legibilidade e honestidade visual |
| Conclusão sem declarar um "vencedor" | As diferenças são de 1 tumor no teste; afirmar superioridade não seria sustentado pelos dados |
| Reexecução completa em kernel limpo | Exigência da especificação, para verificar que os resultados se reproduzem |
| Nenhum ajuste nesta etapa | O teste já foi usado; qualquer mudança agora contaminaria a avaliação |

---

## 6. Interpretação dos resultados obtidos

**Tabela final** (treino = validação cruzada, fora da amostra):

| Modelo | Conjunto | Recall | Precisão | F1 | Acurácia | FN | FP | Malignos |
|---|---|---|---|---|---|---|---|---|
| Dummy | Treino | 0,00% | 0,00% | 0,00% | 62,64% | 170 | 0 | 170 |
| Dummy | Teste | 0,00% | 0,00% | 0,00% | 63,16% | 42 | 0 | 42 |
| SGD - padrão | Treino | 95,88% | 95,88% | 95,88% | 96,92% | 7 | 7 | 170 |
| SGD - padrão | Teste | 92,86% | 86,67% | 89,66% | 92,11% | 3 | 6 | 42 |
| SGD - F2 | Treino | **97,65%** | 92,74% | 95,13% | 96,26% | 4 | 13 | 170 |
| SGD - F2 | Teste | **95,24%** | 86,96% | 90,91% | 92,98% | **2** | 6 | 42 |
| Random Forest - padrão | Treino | 94,71% | 94,15% | 94,43% | 95,82% | 9 | 10 | 170 |
| Random Forest - padrão | Teste | 92,86% | **100,00%** | **96,30%** | **97,37%** | 3 | **0** | 42 |
| Random Forest - F2 | Treino | 96,47% | 92,66% | 94,52% | 95,82% | 6 | 13 | 170 |
| Random Forest - F2 | Teste | 92,86% | 97,50% | 95,12% | 96,49% | 3 | 1 | 42 |

**Interpretação** (mesmos pontos da seção 6.1 do notebook):

1. **Linha de base:** o Dummy tem ≈63% de acurácia e recall 0 no treino e no teste. É a prova de que a acurácia engana neste problema.
2. **Modelos reais:** detectaram 39 ou 40 dos 42 malignos do teste.
3. **Recall, treino × teste:** caiu um pouco em todas as variantes, entre 1,9 p.p. (RF padrão: 94,71% → 92,86%) e 3,6 p.p. (RF F2: 96,47% → 92,86%), o que equivale a 1 ou 2 tumores no teste.
4. **Precisão, treino × teste:** os modelos divergiram. O SGD caiu para ≈87% (6 FP em 72 benignos), e a RF subiu para 97,5–100% (0 ou 1 FP). Com um único teste pequeno, não dá para saber se é variação amostral ou diferença real de generalização.
5. **Limiar F2:** no treino, +3 malignos detectados nos dois modelos, com +6 FP (SGD) e +3 FP (RF). No teste, o SGD ganhou 1 maligno sem FP extra; a RF não ganhou recall e teve +1 FP. O ajuste desloca o equilíbrio, mas o ganho real depende dos dados novos.
6. **Escolha:**
   - prioridade do exercício (recall): **SGD - F2** teve o maior recall no treino (97,65%) e no teste (95,24%);
   - menos falsos positivos e maior acurácia, precisão e F1 no teste: **Random Forest - padrão**;
   - a diferença de recall entre as duas é de **1 tumor**, então não há superioridade estabelecida, e a escolha depende do custo relativo de FN e FP.

**Verificação de reprodutibilidade:**

| Verificação | Resultado |
|---|---|
| Células de código executadas | 55, em sequência (contagem de execução 1 a 55) |
| Erros | Nenhum |
| Avisos nas células (stderr) | Nenhum |
| Saídas de texto comparadas com a execução anterior | **Idênticas em todas as 55 células** |
| Formato do notebook | Validado com `nbformat.validate` |

---

## 7. Erros comuns e limitações

- **Declarar um vencedor com base em 1 tumor:** com 42 malignos no teste, diferenças de 2,4 p.p. no recall não são conclusivas.
- **Comparar só porcentagens:** sem as contagens, 86,67% × 100% de precisão parece enorme, mas são 6 × 0 falsos positivos em 72 benignos. É relevante, mas ainda baseado em poucos casos.
- **Ajustar algo depois de ver o teste:** por exemplo, trocar o limiar da RF. Não foi feito, e não deve ser feito.
- **Esquecer o eixo truncado do gráfico:** começar em 80% amplia as diferenças. O aviso no eixo existe para isso.
- **Limitações do estudo:**
  - dataset pequeno (569 tumores) e um único conjunto de teste;
  - hiperparâmetros padrão, sem ajuste;
  - dados de uma única origem (Wisconsin, 1995);
  - limiares F2 escolhidos com previsões que também aparecem na coluna "Treino" (levemente otimista).
- **Sem validade clínica:** é um exercício acadêmico.

**Possíveis extensões** (não executadas; apenas sugestões para trabalhos futuros):
- validação cruzada repetida (`RepeatedStratifiedKFold`) para estimativas mais estáveis;
- ajuste de hiperparâmetros com validação cruzada aninhada;
- calibração das probabilidades;
- avaliação em dados de outra origem.

---

## 8. Perguntas que o professor poderia fazer

1. **Qual modelo você recomenda?** Pela prioridade do exercício, o SGD com limiar F2, com o maior recall no treino e no teste. Mas a diferença para a RF é de 1 tumor, e a RF teve menos falsos positivos, então não afirmo superioridade.
2. **Por que o recall caiu do treino para o teste?** É uma queda de 1,9 a 3,6 p.p., ou 1 a 2 tumores no teste. É compatível com a variação esperada num teste de 42 malignos.
3. **Por que a precisão do SGD caiu tanto no teste?** Ele classificou 6 dos 72 benignos como malignos. Com um único teste pequeno, não é possível dizer se é acaso ou diferença real de generalização.
4. **O ajuste do limiar valeu a pena?** No SGD, sim: 1 maligno a mais sem custo em FP no teste. Na RF, não houve ganho. O ajuste é útil para expressar a prioridade, mas não garante ganho em todos os dados.
5. **Como você garante que os resultados são reproduzíveis?** Todas as fontes de aleatoriedade usam `random_state=42`, e o notebook foi reexecutado do zero em kernel limpo com saídas idênticas.
6. **Esses resultados poderiam ser usados em um hospital?** Não. É um exercício acadêmico com um dataset pequeno e antigo, sem validação clínica.
7. **O que você faria para melhorar?** Validação cruzada repetida, ajuste de hiperparâmetros com validação aninhada, calibração e testes com dados de outra origem.

---

## 9. Como apresentar oralmente

**Fechamento desta etapa (≈1 minuto):**

> "Para fechar, coloquei lado a lado o desempenho no treino, com previsões fora da amostra, e no teste. O Dummy confirma que a acurácia engana: 63% sem detectar nenhum câncer. O SGD e a Random Forest detectaram 39 ou 40 dos 42 malignos do teste. O recall caiu um pouco do treino para o teste, o equivalente a 1 ou 2 tumores. O ajuste do limiar ajudou o SGD, que ganhou um maligno sem novos alarmes falsos, mas não a Random Forest. O SGD com limiar F2 teve o maior recall nos dois conjuntos, e a Random Forest padrão teve a melhor precisão, sem nenhum falso positivo. Como a diferença é de um tumor, não declaro um vencedor. Reexecutei o notebook do zero e os resultados se reproduziram exatamente. E reforço: é um exercício acadêmico, sem validade clínica."

**Roteiro da apresentação completa** (sugestão de sequência, com o guia de cada parte). Para apresentar, use o notebook **`Exercise/1-Exercise.ipynb`**, que tem as células de código e os títulos das seções, com as mesmas saídas e figuras do notebook de estudo, na mesma ordem:

| Ordem | Parte | Guia | Mostrar |
|---|---|---|---|
| 1 | Problema e dados | 01 | Panorama Geral, `(569, 30)` |
| 2 | Alvo invertido e classes | 02 | `target_names`, tabela de conferência, gráfico 62,7% × 37,3% |
| 3 | Divisão estratificada | 03 | Tabela de proporções |
| 4 | Modelos e Pipeline | 04 | Célula do Pipeline, gráfico antes × depois |
| 5 | Validação cruzada | 05 | Gráfico comparativo, Dummy com recall 0 |
| 6 | *Trade-off* e limiar F2 | 06 | Zoom da curva P×R, tabela FN/FP |
| 7 | Matrizes no treino | 07 | Matriz do Dummy, matrizes do SGD |
| 8 | Teste | 08 | Matrizes do SGD - F2 e da RF padrão, tabela-resumo |
| 9 | Conclusão | 09 | Gráfico treino × teste, seção 6.2 |

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6, matplotlib 3.11.2 e nbconvert (via `uv`). Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**pandas, NumPy e Matplotlib (código desta etapa)**

| Item | Para conferir | Link |
|---|---|---|
| `DataFrame.set_index` | Índice com vários níveis | [pandas — DataFrame.set_index](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html) |
| *MultiIndex* | Como funcionam índices hierárquicos | [pandas — MultiIndex / advanced indexing](https://pandas.pydata.org/docs/user_guide/advanced.html) |
| `unstack` | Passar um nível do índice para colunas | [pandas — DataFrame.unstack](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.unstack.html) |
| `np.arange` | Marcas do eixo Y | [NumPy — numpy.arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html) |
| `ax.set_ylim` / `ax.set_yticks` | Limites e marcas do eixo | [Matplotlib — Axes.set_ylim](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylim.html) · [Axes.set_yticks](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yticks.html) |
| `recall_score` | Métrica central da comparação | [scikit-learn — recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) |

**Reprodutibilidade**

| Item | Para conferir | Link |
|---|---|---|
| `nbconvert --execute` | Executar o notebook do início ao fim em kernel novo | [nbconvert — Executing notebooks](https://nbconvert.readthedocs.io/en/latest/execute_api.html) |
| Uso do `nbconvert` | Opções de linha de comando (`--to notebook`, `--inplace`) | [nbconvert — Using as a command line tool](https://nbconvert.readthedocs.io/en/latest/usage.html) |
| Controle da aleatoriedade | `random_state` no scikit-learn | [scikit-learn — Controlling randomness](https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness) |

**Extensões citadas (não executadas)**

| Item | Para conferir | Link |
|---|---|---|
| `RepeatedStratifiedKFold` | Validação cruzada repetida | [scikit-learn — RepeatedStratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html) |
| Ajuste de hiperparâmetros | `GridSearchCV` e afins | [scikit-learn — Tuning the hyper-parameters](https://scikit-learn.org/stable/modules/grid_search.html) |
| Validação cruzada aninhada | Exemplo oficial | [scikit-learn — Nested vs non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) |
| Calibração de probabilidades | Quando e como calibrar | [scikit-learn — Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) |
