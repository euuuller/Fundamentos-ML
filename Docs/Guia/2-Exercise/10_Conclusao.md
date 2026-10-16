# Etapa 10 — Comparação final, conclusão e notebook de apresentação

Notebook de estudo: `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`, seção **6. Comparação Final e Conclusão** (tabela e gráfico comparativos, 6.1 Interpretação, 6.2 Conclusão).
Notebook de apresentação: **`Exercise/2-Exercise.ipynb`** (gerado nesta etapa).
Correspondência na referência: não há seção equivalente. É o fechamento previsto na especificação, igual à Etapa 9 do Exercício 01.

---

## 1. Objetivo e relação com o exercício

1. Comparar, para cada modelo e limiar, o desempenho **no treino (fora da amostra)** e **no teste**, com todas as métricas do enunciado (acurácia, precisão, recall, F1, AUC) e as contagens de erros.
2. Interpretar o conjunto dos resultados e escrever a **conclusão acadêmica**.
3. **Verificar a reprodutibilidade:** reexecutar o notebook do zero em kernel limpo e comparar as saídas.
4. **Gerar o notebook de apresentação** (código + títulos) e conferir que ele reproduz exatamente as saídas do notebook de estudo.

---

## 2. Código implementado, por blocos

### Bloco 1 — Tabela final treino × teste

```python
y_scores_dummy = cross_val_predict(dummy_clf, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

previsoes_finais = {
    'Dummy': (y_train_pred_dummy, y_scores_dummy, y_pred_dummy, y_scores_teste_dummy),
    'SGD - padrão': (y_train_pred_sgd, y_scores_sgd, y_pred_sgd, y_scores_teste_sgd),
    ...
}
for modelo, (pred_treino, score_treino, pred_teste, score_teste) in previsoes_finais.items():
    for conjunto, y_real, y_pred, y_score in [...]:
        linhas.append({... 'AUC': roc_auc_score(y_real, y_score), 'FN': ..., 'FP': ...})
df_final = pd.DataFrame(linhas).set_index(['Modelo', 'Conjunto'])
```

- Reaproveita **todas as previsões e pontuações já calculadas** (Etapas 7, 8 e 9); nada é treinado ou ajustado.
- A única novidade é a pontuação fora da amostra do Dummy no treino, necessária para calcular a AUC dele (que dá 0,5).
- Em relação ao Exercício 01, a tabela ganhou a coluna **AUC**, que precisa das pontuações e não só das classes.

### Bloco 2 — Gráfico F1 e AUC (treino × teste)

```python
tabela = df_final[metrica].unstack('Conjunto').loc[modelos_reais, [...]]
ax.set_ylim(0.5, 0.97); ax.set_ylabel("Eixo começa em 50% (diferenças ampliadas)")
```

Dois painéis: **F1**, o critério de equilíbrio do exercício, e **AUC**, a métrica que não depende do limiar. O eixo começa em 50% para que as diferenças fiquem visíveis, e o próprio eixo avisa isso.

### Bloco 3 — Interpretação (6.1) e conclusão (6.2), em Markdown

Sete pontos de interpretação e a conclusão com limitações. Todos os números citados vêm da tabela final.

### Bloco 4 — Reprodutibilidade (fora do notebook)

1. Saídas de texto de todas as células de código salvas.
2. Notebook reexecutado em **kernel limpo** (`nbconvert --execute --inplace`).
3. Saídas comparadas célula a célula.

**Observação técnica:** na primeira comparação, 3 células pareciam diferentes. A diferença era **só uma linha em branco**: o Jupyter às vezes divide a mesma saída de texto (stdout) em mais de um bloco, conforme o tempo de execução, e o script de comparação juntava os blocos com quebra de linha. Depois de juntar os blocos de stdout sem separador, a comparação deu **saídas idênticas** em todas as células.

### Bloco 5 — Geração do notebook de apresentação (fora do notebook)

Pela mesma regra do Exercício 01 (descrita no `CLAUDE.md`):
- todas as células de código, na mesma ordem e sem saídas;
- do Markdown, **só os títulos** (cabeçalhos `#` e subtítulos em negrito que antecedem código), sem textos explicativos nem "Observações".

Em seguida, o notebook gerado foi executado e suas saídas comparadas com as do notebook de estudo.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `y_scores_dummy` | Pontuação fora da amostra do Dummy no treino (constante) |
| `previsoes_finais` | Para cada variante: previsões e pontuações no treino e no teste |
| `df_final` | Tabela final com índice (Modelo, Conjunto) |
| `roc_auc_score(y_real, y_score)` | AUC a partir das pontuações |
| `set_index([...])` / `unstack('Conjunto')` | Índice de dois níveis / passar "Conjunto" para colunas |
| `ax.set_yticks(np.arange(0.50, 0.901, 0.05))` | Marcas do eixo Y até 90% |
| `modelos_reais` | As 4 variantes de SGD/RF exibidas no gráfico |

---

## 4. Conceitos e fórmulas

**Treino (fora da amostra) × teste:** as duas estimativas usam modelos que não viram os passageiros avaliados, mas:
- o treino tem 712 passageiros (273 sobreviventes) e o teste, 179 (69). No teste, cada sobrevivente vale $1/69 \approx 1{,}45$ p.p. de recall, contra $1/273 \approx 0{,}37$ p.p. no treino;
- os limiares F1 foram escolhidos com as previsões do treino (levemente otimista); o teste não participou de nenhuma decisão.

**Estabilidade:** um modelo cujo desempenho varia muito entre as dobras da validação cruzada (como o SGD aqui) pode ter, num único teste, resultado bem acima ou abaixo da sua média. Por isso, a validação cruzada (5 avaliações) e o teste (1 avaliação) devem ser lidos **juntos**.

**Reprodutibilidade:** com `random_state=42` em todas as fontes de aleatoriedade (divisão, dobras, SGD, RF), o notebook produz exatamente os mesmos números a cada execução.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Treino e teste lado a lado, com AUC e contagens | Mostra se o comportamento estimado se confirmou; as contagens evitam superestimar diferenças pequenas |
| F1 e AUC no gráfico | F1 é o critério de equilíbrio adotado; AUC independe do limiar |
| Eixo a partir de 50% com aviso | Legibilidade com honestidade visual (mesma solução do Exercício 01) |
| Recomendar a RF com limiar F1, com ressalvas | Foi a melhor no teste e a mais consistente na validação cruzada, mas as diferenças são de poucos passageiros |
| Nenhum ajuste nesta etapa | O teste já foi usado |
| Comparação de saídas que junta os blocos de stdout | A divisão em blocos é um detalhe de execução, não uma diferença de resultado |
| Notebook de apresentação gerado, não editado à mão | Garante que o código apresentado é exatamente o do notebook de estudo |

---

## 6. Interpretação dos resultados obtidos

**Tabela final:**

| Modelo | Conjunto | Acurácia | Precisão | Recall | F1 | AUC | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Dummy | Treino | 61,66% | 0,00% | 0,00% | 0,00% | 50,00% | 273 | 0 |
| Dummy | Teste | 61,45% | 0,00% | 0,00% | 0,00% | 50,00% | 69 | 0 |
| SGD - padrão | Treino | 71,91% | 61,89% | 69,60% | 65,52% | 78,68% | 83 | 117 |
| SGD - padrão | Teste | 79,33% | 78,57% | 63,77% | 70,40% | 83,70% | 25 | 12 |
| SGD - F1 | Treino | 72,05% | 60,57% | 77,66% | 68,06% | 78,68% | 61 | 138 |
| SGD - F1 | Teste | 75,98% | 67,57% | 72,46% | 69,93% | 83,70% | 19 | 24 |
| RF - padrão | Treino | 78,51% | 70,83% | 74,73% | 72,73% | 86,23% | 69 | 84 |
| RF - padrão | Teste | 81,01% | 76,92% | 72,46% | 74,63% | 83,04% | 19 | 15 |
| RF - F1 | Treino | 80,76% | 76,36% | 72,16% | 74,20% | 86,23% | 76 | 61 |
| RF - F1 | Teste | **82,12%** | **80,33%** | 71,01% | **75,38%** | 83,04% | 20 | 12 |

**Interpretação** (os mesmos pontos da seção 6.1 do notebook):

1. **Linha de base:** o Dummy tem ≈61% de acurácia, recall 0 e AUC 0,5 no treino e no teste. A acurácia sozinha engana.
2. **Modelos reais:** identificaram 44 a 50 dos 69 sobreviventes do teste (F1 de 69,93% a 75,38%; AUC acima de 83%).
3. **Random Forest estável:** o F1 ficou parecido no treino e no teste (72,73% → 74,63%; 74,20% → 75,38%). A AUC caiu de 86,23% para 83,04%.
4. **SGD instável:** foi bem melhor no teste do que no treino (F1 65,52% → 70,40%; AUC 78,68% → 83,70%), coerente com a grande variação entre dobras vista na Etapa 6. No teste, empatou em AUC com a RF, mas um único teste pequeno não basta para concluir que se equivalem.
5. **Limiar F1:** na RF, melhorou o F1 no treino **e** no teste; no SGD, melhorou no treino mas **não** no teste (69,93% × 70,40%, com FP dobrando).
6. **Escolha:** a **Random Forest com limiar F1** foi a melhor no teste (F1 75,38%, acurácia 82,12%, precisão 80,33%, 32 erros) e a mais consistente na validação cruzada.
7. **Associações:** sexo e classe estão fortemente associados à sobrevivência nos dados. Os modelos usam essas associações, mas o exercício **não explica causas**.

**Comparação com o Exercício 01:** no câncer, os modelos chegaram a F1 ≈ 90–96% e não houve vencedor claro. No Titanic, o problema é mais difícil (F1 ≈ 70–75%), mas a Random Forest mostrou **vantagem consistente** na validação cruzada. Nos dois exercícios, o ajuste do limiar funcionou no treino e só parcialmente no teste: em dados novos, o ganho do ajuste não é garantido.

**Reprodutibilidade e notebooks:**

| Verificação | Resultado |
|---|---|
| Notebook de estudo: células | 154 (68 de código) |
| Execução em kernel limpo | 68 células em sequência (contagem 1 a 68), 0 erros, 0 avisos nas células |
| Comparação entre execuções | **Saídas idênticas** nas 68 células (após juntar os blocos de stdout) |
| Notebook de apresentação | `Exercise/2-Exercise.ipynb`: 114 células (68 de código + 46 títulos) |
| Apresentação × estudo | **Saídas idênticas** nas 68 células; 20 figuras; 0 erros; 0 avisos |
| Pasta de dados | O caminho `../datasets` funciona nos dois notebooks (mesma profundidade); nenhuma pasta extra criada |

---

## 7. Erros comuns e limitações

- **Declarar o SGD equivalente ou superior pela AUC do teste:** contraria a validação cruzada e se baseia num único teste de 179 passageiros.
- **Tratar associações como causas** ("ser da 1ª classe faz sobreviver"): os dados não permitem essa conclusão.
- **Ajustar algo agora** (por exemplo, voltar o limiar do SGD para 0): seria usar o teste para decidir.
- **Comparar saídas de notebooks sem considerar a divisão do stdout:** gera falsos alarmes de "não reprodutível", como foi identificado e corrigido nesta etapa.
- **Limitações do estudo:**
  - dados: 891 passageiros, 69 sobreviventes no teste, um único *split* e dados históricos de um único evento;
  - modelagem: hiperparâmetros padrão, imputação simples (mediana/moda) e nenhuma engenharia de atributos.

**Possíveis extensões** (não executadas):
- atributos derivados (título extraído do nome, tamanho da família);
- imputação mais elaborada (`IterativeImputer`);
- ajuste de hiperparâmetros com validação cruzada;
- validação cruzada repetida.

---

## 8. Perguntas que o professor poderia fazer

1. **Qual modelo você recomenda?** A Random Forest com limiar F1: melhor F1, acurácia e precisão no teste e mais consistente na validação cruzada.
2. **Por que o SGD foi melhor no teste do que no treino?** Ele é instável: na validação cruzada, o F1 variou de 56% a 71% entre as dobras. O teste, com 179 passageiros, caiu num caso favorável. Não é evidência de que ele generalize melhor.
3. **O ajuste do limiar valeu a pena?** Para a RF, sim (melhorou no treino e no teste). Para o SGD, só no treino.
4. **Os resultados são reproduzíveis?** Sim: com `random_state=42`, a reexecução em kernel limpo produziu saídas idênticas; o notebook de apresentação também reproduz as mesmas saídas.
5. **O que diferenciou este exercício do primeiro?** Dados com ausentes e texto (exigiram `ColumnTransformer` no Pipeline), o limiar por F1 em vez de F2, a curva ROC/AUC e uma vantagem consistente de um modelo.
6. **Pode-se dizer que o sexo causou a sobrevivência?** Não. Os dados mostram associação; causas exigiriam outro tipo de estudo.
7. **O que você faria para melhorar?** Engenharia de atributos (título, família), imputação mais elaborada, ajuste de hiperparâmetros e validação cruzada repetida.

---

## 9. Como apresentar oralmente

**Fechamento (≈1 minuto):**

> "Para fechar, comparei treino e teste. O Dummy confirma que a acurácia engana: 61% sem identificar nenhum sobrevivente. A Random Forest foi estável: F1 em torno de 73 a 75% nos dois conjuntos. Com o limiar escolhido pelo F1, ela foi a melhor no teste: 82% de acurácia, F1 de 75% e só 32 erros. O SGD foi instável: no teste ficou melhor do que na validação cruzada e empatou em AUC com a Random Forest, mas isso é coerente com a variação que ele já mostrava; um teste pequeno não basta para dizer que os dois se equivalem. O ajuste do limiar ajudou a Random Forest no treino e no teste, e o SGD só no treino. Reexecutei tudo do zero e os resultados se reproduziram exatamente. E as relações que encontramos, como sexo e classe, são associações nos dados, não causas."

**Roteiro da apresentação completa.** Use o notebook **`Exercise/2-Exercise.ipynb`**, que tem o código e os títulos das seções, com as mesmas saídas do notebook de estudo:

| Ordem | Parte | Guia | Mostrar |
|---|---|---|---|
| 1 | Problema e dados (`test.csv` sem rótulo) | 01 | Panorama Geral, `(1309, 12)` → `(891, 12)` |
| 2 | Alvo, ausentes e associações | 02 | Tabela de ausentes, taxas por sexo e classe |
| 3 | Divisão estratificada | 03 | Tabela de proporções e de ausentes por parte |
| 4 | Pré-processamento | 04 | `ColumnTransformer`, exemplo antes × depois |
| 5 | Modelos em Pipeline | 05 | Pipeline do SGD, pesos, exemplo atípico |
| 6 | Validação cruzada com AUC | 06 | Gráfico comparativo, tabela por dobra |
| 7 | ROC e limiar F1 | 07 | Curvas ROC, tabela FN/FP (limiares em direções opostas) |
| 8 | Matrizes no treino | 08 | Matriz do Dummy, matrizes da RF |
| 9 | Teste | 09 | Curvas ROC no teste, tabela-resumo |
| 10 | Conclusão | 10 | Gráfico F1/AUC treino × teste, seção 6.2 |

---

## 10. Documentação oficial

Links para a versão *stable* ou atual de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6 e matplotlib 3.11.2. Se a versão do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Código desta etapa**

| Item | Para conferir | Link |
|---|---|---|
| `roc_auc_score` | AUC no treino e no teste | [scikit-learn — roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) |
| `cross_val_predict` | Pontuação fora da amostra do Dummy | [scikit-learn — cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) |
| `DataFrame.set_index` | Índice de dois níveis | [pandas — DataFrame.set_index](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html) |
| `unstack` | Nível do índice → colunas | [pandas — DataFrame.unstack](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.unstack.html) |
| `ax.set_yticks` | Marcas do eixo Y | [Matplotlib — Axes.set_yticks](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yticks.html) |

**Reprodutibilidade e notebooks**

| Item | Para conferir | Link |
|---|---|---|
| Executar notebooks | `nbconvert --execute` em kernel novo | [nbconvert — Executing notebooks](https://nbconvert.readthedocs.io/en/latest/execute_api.html) |
| Linha de comando do `nbconvert` | `--to notebook --inplace` | [nbconvert — Command line usage](https://nbconvert.readthedocs.io/en/latest/usage.html) |
| `nbformat` | Ler/gravar notebooks (geração da apresentação) | [nbformat — Python API](https://nbformat.readthedocs.io/en/latest/api.html) |
| Mensagens de *stream* | Por que o stdout pode chegar em vários blocos | [Jupyter — Streams (stdout, stderr, etc)](https://jupyter-client.readthedocs.io/en/latest/messaging.html#streams-stdout-stderr-etc) |
| Controle da aleatoriedade | `random_state` | [scikit-learn — Controlling randomness](https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness) |

**Extensões citadas (não executadas)**

| Item | Para conferir | Link |
|---|---|---|
| `IterativeImputer` | Imputação multivariada | [scikit-learn — Iterative imputer](https://scikit-learn.org/stable/modules/impute.html#iterative-imputer) |
| Ajuste de hiperparâmetros | `GridSearchCV` e afins | [scikit-learn — Tuning the hyper-parameters](https://scikit-learn.org/stable/modules/grid_search.html) |
| `RepeatedStratifiedKFold` | Validação cruzada repetida | [scikit-learn — RepeatedStratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html) |
| Exemplo oficial (Titanic) | Pipeline com dados mistos e ideias de extensão | [scikit-learn — Column Transformer with Mixed Types](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html) |
