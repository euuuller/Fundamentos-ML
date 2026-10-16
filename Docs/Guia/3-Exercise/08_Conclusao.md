# Etapa 8 — Comparação final, conclusão e notebook de apresentação

Notebook de estudo: `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`, seção **6. Comparação Final e Conclusão** (tabela e gráfico comparativos, 6.1 Interpretação, 6.2 Conclusão).
Notebook de apresentação: **`Exercise/3-Exercise.ipynb`** (gerado nesta etapa).
Correspondência na referência: não há seção equivalente (a referência termina no teste). É o fechamento previsto na especificação, como nos Exercícios 01 e 02.

---

## 1. Objetivo e relação com o exercício

1. Comparar, para os 4 modelos, o desempenho no **treino (fora da amostra)** e no **teste**, com acurácia, métricas macro e número de erros.
2. Interpretar o conjunto e escrever a **conclusão acadêmica**.
3. **Verificar a reprodutibilidade:** reexecutar o notebook do zero, duas vezes, e comparar as saídas.
4. **Gerar o notebook de apresentação** (código + títulos) e conferir que ele reproduz as saídas do notebook de estudo.

---

## 2. Código implementado, por blocos

### Bloco 1 — Tabela final treino × teste

```python
previsoes_finais = {
    nome: (previsoes_treino[nome], y_pred_teste)
    for nome, y_pred_teste in [('Dummy Classifier', y_pred_dummy), ('SGD Classifier', y_pred_sgd),
                               ('Random Forest', y_pred_forest), ('SVC (RBF)', y_pred_svc)]
}
...
df_final = pd.DataFrame(linhas).set_index(['Modelo', 'Conjunto'])
```

Reaproveita as previsões fora da amostra da Etapa 6 (`previsoes_treino`) e as previsões do teste da Etapa 7: **nada é treinado nem ajustado**. Para cada modelo e conjunto, calcula acurácia, precisão, recall e F1 macro, o número de erros e o número de amostras.

**Observação:** as métricas do treino aqui são calculadas sobre as 142 previsões **agrupadas**. Na Etapa 5, eram a **média das 5 dobras**. Por isso há diferenças mínimas (por exemplo, F1 macro do SGD: 99,27% agrupado contra 99,29% na média por dobra).

### Bloco 2 — Gráfico F1 macro e acurácia (treino × teste)

```python
tabela = df_final[metrica].unstack('Conjunto').loc[modelos_reais, [...]]
ax.set_ylim(0.85, 1.04); ax.set_ylabel("Eixo começa em 85% (diferenças ampliadas)")
fig.legend(handles, labels, loc='lower center', ncol=2)
```

Dois painéis para os 3 modelos reais (o Dummy fica na tabela, porque seus ≈40% achatariam a escala). O eixo começa em 85% para tornar visíveis diferenças de poucos pontos, e o próprio eixo avisa isso. **Correção durante a etapa:** a legenda de cada painel cobria a base das barras; foi substituída por uma legenda única abaixo dos gráficos.

### Bloco 3 — Interpretação (6.1) e conclusão (6.2), em Markdown

Seis pontos de interpretação e a conclusão com limitações. Todos os números vêm da tabela final.

### Bloco 4 — Reprodutibilidade (fora do notebook)

Duas execuções completas em kernel limpo (`nbconvert --execute --inplace`), com comparação das saídas de texto célula a célula (juntando os blocos de stdout, como registrado no `CLAUDE.md`).

### Bloco 5 — Notebook de apresentação (fora do notebook)

Gerado pela regra do `CLAUDE.md`: todas as células de código e só os títulos do Markdown. Depois foi executado e teve as saídas comparadas com as do notebook de estudo.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `previsoes_finais` | Modelo → (previsões do treino fora da amostra, previsões do teste) |
| `df_final` | Tabela (Modelo, Conjunto) com métricas, erros e amostras |
| `unstack('Conjunto')` | Treino e teste em colunas, para o gráfico agrupado |
| `fig.legend(...)` + `tight_layout(rect=...)` | Legenda única abaixo dos painéis, sem sobrepor as barras |
| `modelos_reais` | SGD, Random Forest e SVC |

---

## 4. Conceitos e fórmulas

**Treino (fora da amostra) × teste:** as duas estimativas usam modelos que não viram as amostras avaliadas, mas o treino tem 142 amostras e o teste 36. A granularidade é diferente:

$$\Delta Acc_{treino} = \frac{1}{142} \approx 0{,}7 \text{ p.p.} \qquad \Delta Acc_{teste} = \frac{1}{36} \approx 2{,}8 \text{ p.p.}$$

**Variação entre divisões:** quando os modelos erram 0 a 3 amostras, mudar quais amostras caem no teste pode inverter a ordem entre eles. É o que aconteceu aqui. A forma honesta de comparar é olhar o **conjunto** das evidências: as 5 dobras da validação cruzada e o teste.

**Agregado treino + teste:** somando os erros dos dois conjuntos (178 amostras), cada modelo real errou **3**. É uma leitura descritiva, que mistura uma estimativa fora da amostra com o teste, mas mostra que as diferenças observadas em cada conjunto se compensam.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Comparar treino (fora da amostra) e teste com erros absolutos | Com conjuntos pequenos, contar erros evita superestimar diferenças percentuais |
| Previsões agrupadas no treino | Permitem contar erros e calcular as métricas do mesmo jeito que no teste |
| Gráfico só com os modelos reais e eixo a partir de 85% | Legibilidade; o aviso no eixo evita exagerar as diferenças |
| Não eleger um vencedor | A ordem inverteu entre treino e teste, e cada modelo errou 3 amostras no total |
| Duas execuções em kernel limpo | Exigência da especificação (reprodutibilidade) |
| Notebook de apresentação gerado, não editado | Garante o mesmo código e as mesmas saídas do notebook de estudo |

---

## 6. Interpretação dos resultados obtidos

**Tabela final:**

| Modelo | Conjunto | Acurácia | Precisão (macro) | Recall (macro) | F1 (macro) | Erros |
|---|---|---|---|---|---|---|
| Dummy | Treino | 40,14% | 13,38% | 33,33% | 19,10% | 85 / 142 |
| Dummy | Teste | 38,89% | 12,96% | 33,33% | 18,67% | 22 / 36 |
| SGD | Treino | **99,30%** | 99,15% | 99,42% | **99,27%** | **1** / 142 |
| SGD | Teste | 94,44% | 95,05% | 94,29% | 94,53% | 2 / 36 |
| Random Forest | Treino | 97,89% | 97,64% | 98,25% | 97,89% | 3 / 142 |
| Random Forest | Teste | **100,00%** | **100,00%** | **100,00%** | **100,00%** | **0** / 36 |
| SVC (RBF) | Treino | 98,59% | 98,56% | 98,71% | 98,62% | 2 / 142 |
| SVC (RBF) | Teste | 97,22% | 97,78% | 96,67% | 97,10% | 1 / 36 |

**Interpretação** (os mesmos pontos da seção 6.1 do notebook):

1. **Linha de base:** o Dummy tem ≈39–40% de acurácia, recall macro de 1/3 e F1 macro de ≈19% nos dois conjuntos.
2. **Desempenho muito alto:** F1 macro de 97,89% a 99,27% no treino e de 94,53% a 100% no teste.
3. **Ordem invertida:** no treino, SGD (1 erro) < SVC (2) < RF (3); no teste, RF (0) < SVC (1) < SGD (2). Somando, **3 erros em 178 para cada modelo**. Não há modelo claramente superior.
4. **Estratégias multiclasse:** OvR linear (SGD), OvO com kernel (SVC) e o modelo nativo (RF) chegaram a resultados equivalentes. O SGD linear quase perfeito indica cultivares quase separáveis por fronteiras lineares depois da padronização. No MNIST da referência, o SGD ficava claramente atrás (≈90% × ≈96%).
5. **Fronteira difícil:** 6 dos 9 erros dos modelos reais (treino + teste) envolveram a confusão entre `class_1` e `class_2`, coerente com a sobreposição vista na análise exploratória.
6. **Médias:** macro, ponderada e micro coincidem nos modelos reais e divergem só no Dummy.

**Comparação com os exercícios anteriores:**

| Exercício | Tarefa | Melhor F1 no teste | Vencedor claro? |
|---|---|---|---|
| 01 — Câncer | Binária (recall) | ≈96% (RF) | Não (diferença de 1 tumor) |
| 02 — Titanic | Binária (F1, ROC) | ≈75% (RF com limiar F1) | RF mais consistente na validação cruzada |
| 03 — Wine | Multiclasse (macro) | 100% (RF) | Não (3 erros em 178 para cada modelo) |

**Reprodutibilidade e notebooks:**

| Verificação | Resultado |
|---|---|
| Notebook de estudo | 113 células (53 de código) |
| Duas execuções em kernel limpo | Contagem de execução 1 a 53, 0 erros, 0 avisos; **saídas idênticas** nas 53 células |
| Notebook de apresentação | `Exercise/3-Exercise.ipynb`: 88 células (53 de código + 35 títulos) |
| Apresentação × estudo | **Saídas idênticas** nas 53 células; 14 figuras; 0 erros; 0 avisos |

---

## 7. Erros comuns e limitações

- **Eleger o melhor modelo pelo teste** (RF com 100%): a vantagem é de 1–2 amostras e não se repetiu na validação cruzada.
- **Achar que 100% no teste significa modelo perfeito:** com 36 amostras, isso é compatível com a taxa de erro de ≈2% vista na validação cruzada.
- **Ignorar a estrutura dos erros:** as métricas agregadas escondem que quase todos os erros estão entre `class_1` e `class_2`.
- **Esquecer o eixo truncado do gráfico:** começar em 85% amplia visualmente as diferenças.
- **Limitações do estudo:** 178 amostras; teste de 36; um único *split*; hiperparâmetros padrão; dados de uma única região.

**Possíveis extensões** (não executadas):
- validação cruzada repetida (`RepeatedStratifiedKFold`) para estimativas mais estáveis;
- ajuste de hiperparâmetros com validação cruzada;
- comparar com a Regressão Logística multinomial (outro modelo nativo, citado na referência).

---

## 8. Perguntas que o professor poderia fazer

1. **Qual modelo você recomenda?** Os três modelos reais são equivalentes neste problema: cada um errou 3 das 178 amostras somando treino e teste. Qualquer um seria adequado; a escolha poderia considerar simplicidade (SGD linear) ou custo computacional.
2. **Por que a Random Forest acertou 100% no teste e foi a pior na validação cruzada?** Porque as diferenças são de 1 a 3 amostras, e quais amostras caem em cada conjunto muda a ordem entre os modelos.
3. **OvR ou OvO fez diferença?** Não aqui: com 3 classes bem separadas, as três abordagens tiveram desempenho equivalente.
4. **Onde os modelos erram?** Principalmente na fronteira entre `class_1` e `class_2` (6 de 9 erros).
5. **Os resultados são reproduzíveis?** Sim: duas execuções do zero deram saídas idênticas, e o notebook de apresentação reproduz exatamente as mesmas saídas.
6. **O que você faria para melhorar a confiabilidade?** Validação cruzada repetida e, com mais dados, um conjunto de teste maior.

---

## 9. Como apresentar oralmente

**Fechamento (≈1 minuto):**

> "Para fechar, comparei treino, com previsões fora da amostra, e teste. O Dummy acerta cerca de 40%, só a cultivar mais comum. Os três modelos reais ficaram acima de 94% de F1 macro no teste e acima de 97% na validação cruzada. O ponto principal é que a ordem entre eles inverteu: no treino o SGD errou menos, no teste a Random Forest acertou tudo. Somando treino e teste, cada um errou exatamente 3 de 178 vinhos. Então não há vencedor: as três estratégias, Um-contra-o-Resto, Um-contra-Um e o modelo nativo, funcionam igualmente bem aqui, porque as cultivares são bem separáveis pelos atributos químicos. Os poucos erros aparecem quase sempre entre a `class_1` e a `class_2`, como a análise exploratória sugeria. Reexecutei tudo do zero, e os resultados se reproduziram exatamente."

**Roteiro da apresentação completa.** Use o notebook **`Exercise/3-Exercise.ipynb`**, que tem o código e os títulos das seções, com as mesmas saídas do notebook de estudo:

| Ordem | Parte | Guia | Mostrar |
|---|---|---|---|
| 1 | Problema e dados | 01 | Panorama Geral, `(178, 13)` |
| 2 | Alvo, ordem das amostras e escalas | 02 | Gráfico da ordem por classe, tabela de médias por classe, boxplots |
| 3 | Divisão estratificada e embaralhada | 03 | Rótulos embaralhados, tabela de proporções |
| 4 | Estratégias multiclasse e modelos | 04 | Tabela OvR × OvO, pontuações da amostra (SGD, RF, SVC) |
| 5 | Validação cruzada (macro) | 05 | Gráfico-resumo, tabela macro × ponderada × micro |
| 6 | Matrizes no treino e análise por classe | 06 | Figura do SGD (3 versões), pares de confusão |
| 7 | Teste | 07 | Tabela-resumo com erros, matrizes relativas |
| 8 | Conclusão | 08 | Gráfico treino × teste, seção 6.2 |

---

## 10. Documentação oficial

Links para a versão *stable* ou atual de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, pandas 3.0.6 e matplotlib 3.11.2. Se a versão do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Código desta etapa**

| Item | Para conferir | Link |
|---|---|---|
| `accuracy_score` / `f1_score` | Métricas da tabela final | [scikit-learn — accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) · [f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) |
| `DataFrame.set_index` | Índice (Modelo, Conjunto) | [pandas — DataFrame.set_index](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html) |
| `unstack` | Conjunto → colunas | [pandas — DataFrame.unstack](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.unstack.html) |
| `Figure.legend` | Legenda única da figura | [Matplotlib — Figure.legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.legend.html) |
| `get_legend_handles_labels` | Reaproveitar as entradas da legenda | [Matplotlib — Axes.get_legend_handles_labels](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_legend_handles_labels.html) |
| `tight_layout(rect=...)` | Reservar espaço para a legenda | [Matplotlib — pyplot.tight_layout](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tight_layout.html) |

**Reprodutibilidade e notebooks**

| Item | Para conferir | Link |
|---|---|---|
| Executar notebooks | `nbconvert --execute` em kernel novo | [nbconvert — Executing notebooks](https://nbconvert.readthedocs.io/en/latest/execute_api.html) |
| `nbformat` | Geração do notebook de apresentação | [nbformat — Python API](https://nbformat.readthedocs.io/en/latest/api.html) |
| Mensagens de *stream* | Por que o stdout pode chegar em blocos | [Jupyter — Streams (stdout, stderr, etc)](https://jupyter-client.readthedocs.io/en/latest/messaging.html#streams-stdout-stderr-etc) |
| Controle da aleatoriedade | `random_state` | [scikit-learn — Controlling randomness](https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness) |

**Extensões citadas (não executadas)**

| Item | Para conferir | Link |
|---|---|---|
| `RepeatedStratifiedKFold` | Validação cruzada repetida | [scikit-learn — RepeatedStratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html) |
| Ajuste de hiperparâmetros | `GridSearchCV` e afins | [scikit-learn — Tuning the hyper-parameters](https://scikit-learn.org/stable/modules/grid_search.html) |
| Estratégias multiclasse | Revisão (modelos nativos, OvR, OvO) | [scikit-learn — Multiclass and multioutput algorithms](https://scikit-learn.org/stable/modules/multiclass.html) |
