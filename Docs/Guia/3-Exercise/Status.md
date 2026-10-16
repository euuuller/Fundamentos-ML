# Status — Exercício 03: Classificação Multiclasse de Vinhos (Wine)

Última atualização: 2026-10-12

## Andamento

| Etapa | Descrição | Guia | Situação |
|---|---|---|---|
| 0 | Análise da referência e planejamento | [`00_Plano.md`](00_Plano.md) | ✅ Concluída |
| 1 | Importações e obtenção dos dados | [`01_Importacoes_e_Dados.md`](01_Importacoes_e_Dados.md) | ✅ Concluída |
| 2 | Inspeção inicial, definição do alvo e EDA simples | [`02_Inspecao_e_Analise_Exploratoria.md`](02_Inspecao_e_Analise_Exploratoria.md) | ✅ Concluída |
| 3 | Separação dos dados de treino e teste | [`03_Separacao_Treino_Teste.md`](03_Separacao_Treino_Teste.md) | ✅ Concluída |
| 4 | Estratégias multiclasse e treinamento dos quatro modelos | [`04_Estrategias_e_Treinamento.md`](04_Estrategias_e_Treinamento.md) | ✅ Concluída |
| 5 | Validação cruzada e comparação das métricas macro | [`05_Validacao_Cruzada.md`](05_Validacao_Cruzada.md) | ✅ Concluída |
| 6 | Matrizes de confusão no treino (fora da amostra) e análise por classe | [`06_Matrizes_Confusao_Treino.md`](06_Matrizes_Confusao_Treino.md) | ✅ Concluída |
| 7 | Avaliação final no teste | [`07_Avaliacao_Teste.md`](07_Avaliacao_Teste.md) | ✅ Concluída |
| 8 | Comparação final, conclusão e notebook de apresentação | [`08_Conclusao.md`](08_Conclusao.md) | ✅ Concluída |

**Exercício 03 concluído.**
- Notebook de estudo: `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb` (113 células, 53 de código), reproduzido duas vezes em kernel limpo com saídas idênticas.
- Notebook de apresentação: `Exercise/3-Exercise.ipynb` (88 células: 53 de código + 35 títulos), com saídas idênticas às do notebook de estudo.
- Guias: `Docs/Guia/3-Exercise/`. Especificação: `Docs/Prompt/3-Prompt.md`.

## Registro da Etapa 0

- Referência `Docs/reference/2-Arquivo.ipynb` lida por completo (75 células): MNIST multiclasse (SGD/OvR, Dummy, RF, SVC/OvO; CV com métricas macro; 3 matrizes por modelo; teste) + enunciado do Exercício 3.
- 11 problemas da referência identificados (ver `00_Plano.md` § 1.6) e as adaptações correspondentes definidas (§ 1.7).
- Dados conferidos no ambiente: `load_wine()` 178 × 13, sem ausentes, alvo 59/71/48, **amostras ordenadas por classe**; teste de 20% = 36 amostras.
- Decisões (já registradas no `3-Prompt.md`): os 4 modelos da referência; métricas macro + relatório por classe; 3 tipos de matriz no treino (fora da amostra) + relativa no teste; sem ROC e sem ajuste de limiar.
- Nenhum notebook criado, nenhum treino executado, nenhum commit.
- Skills: nenhuma (justificativa no `00_Plano.md` § 4).

## Registro da Etapa 1

- Notebook `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb` criado: título, "1. Observar o Panorama Geral" (com a nota sobre métricas macro e a ausência de ROC/limiar) e "2. Obtenção dos Dados" (início). São 10 células, 4 de código.
- `plt.rc` configurado uma única vez. `wine = load_wine()`; `X = pd.DataFrame(wine.data, columns=wine.feature_names)` → (178, 13); `y = wine.target` → (178,); 13 atributos listados.
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis para as próximas etapas: `wine`, `X`, `y`.

## Registro da Etapa 2

- Seções "2.1. Definição do Alvo" e "2.2. Descrição dos Dados" acrescentadas (15 células, 8 de código); o notebook tem agora 25 células.
- Codificação conferida: 0/1/2 = `class_0`/`class_1`/`class_2` (**sem recodificação**); `nomes_classes` criado a partir de `target_names`.
- Distribuição: 59 (33,15%), 71 (39,89%), 48 (26,97%), levemente desbalanceada.
- **Ordem por classe** evidenciada: a classe muda nas posições 59 e 130 (três blocos), com gráfico. Toda divisão deve embaralhar.
- Sem nulos. Escalas: `proline` 278–1680 (desvio 314,9) × `nonflavanoid_phenols` 0,13–0,66 (desvio 0,124).
- Médias por classe e diferença em desvios: maiores em `flavanoids` (2,20), `od280/od315` (2,08), `proline` (1,89); menores em `magnesium` (0,83) e `ash` (0,77). Boxplots de `flavanoids`, `proline` e `color_intensity` (escolhidos pela tabela; cada um destaca uma cultivar).
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis para as próximas etapas: `nomes_classes`, `serie_y`.

## Registro da Etapa 3

- Seção "2.3. Separação dos Dados de Treino e Teste" acrescentada (5 células, 2 de código); o notebook tem agora 30 células.
- `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` (embaralhamento padrão): treino (142, 13) com 47/57/38 e teste (36, 13) com 12/14/10.
- Proporções: completo 33,15/39,89/26,97%; treino 33,10/40,14/26,76%; teste 33,33/38,89/27,78%.
- Embaralhamento evidenciado (primeiros rótulos do treino misturados).
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- `X_test` e `y_test` estão **reservados para a Etapa 7**.

## Registro da Etapa 4

- Seção "3. Classificação Multiclasse" acrescentada (20 células, 9 de código); o notebook tem agora 50 células.
- Modelos treinados só com `X_train`/`y_train`:
  - `sgd_clf = make_pipeline(StandardScaler(), SGDClassifier(class_weight='balanced', max_iter=1000, tol=1e-3, random_state=42))`;
  - `dummy_clf = DummyClassifier(strategy="most_frequent")`;
  - `forest_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`;
  - `svc_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale', random_state=42))`.
- Amostra de exemplo (posição 0 do treino, `class_0`): SGD → class_0 (pontuações 35,55 / −57,17 / −97,19); Dummy → class_1; RF → class_0 (0,98 dos votos); SVC → class_0 (2,23 / 0,91 / −0,22, isto é, 2/1/0 duelos).
- Classificadores internos: SGD 3 (`coef_.shape[0]`); SVC 3 (N(N−1)/2 com `len(classes_)`).
- Verificação auxiliar (fora do notebook): duelos OvO do SVC para a amostra: 0×1 → 0; 0×2 → 0; 1×2 → 1, com votos [2, 1, 0], consistentes com as pontuações agregadas.
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis para as próximas etapas: `sgd_clf`, `dummy_clf`, `forest_clf`, `svc_clf`.

## Registro da Etapa 5

- Seções "4. Avaliação de Performance" (métricas por classe e formas de média), "4.1 Validação Cruzada" e "4.2 Gráfico Resumo" acrescentadas (21 células, 9 de código); o notebook tem agora 71 células.
- `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`; métricas `accuracy`, `precision_macro` e `f1_macro` (com `zero_division=0`), `recall_macro`, e ainda `f1_weighted`/`f1_micro` (só para comparação). Sem avisos.
- F1 macro (média ± desvio): Dummy 19,09% ± 0,35; **SGD 99,29% ± 1,42**; RF 97,89% ± 2,85; SVC 98,65% ± 1,67. Acurácia do Dummy 40,12%; recall macro do Dummy 33,33%.
- As diferenças entre SGD, SVC e RF (≈1 a 2 amostras) são menores que a variação entre dobras; cada modelo tem dobras com 100% e dobras com erro. Não há superioridade clara.
- Macro × ponderado × micro: coincidem nos modelos reais; no Dummy, 19,09% × 22,99% × 40,12% (micro = acurácia).
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis para as próximas etapas: `cv`, `metricas`, `nomes_metricas`, `resultados_*`, `resultados_cv`, `df_comparacao`.

## Registro da Etapa 6

- Seções "4.3 Matrizes de Confusão (Treino, fora da amostra)" e "4.4 Análise por Classe" acrescentadas (16 células, 8 de código); o notebook tem agora 87 células.
- Função `plot_matrizes_confusao(y_real, y_pred, nome_modelo, conjunto)`: 3 versões (absoluta, relativa, somente erros) lado a lado, por modelo. Usada no treino; na Etapa 7 (teste), a decisão é só a matriz relativa, feita com uma célula simples por modelo (o guia 06 foi corrigido nesse ponto).
- Previsões fora da amostra (`cross_val_predict`, mesmo `cv`). Erros: SGD **1**, SVC 2, RF 3, Dummy 85 (de 142).
- Por classe: `class_1` concentra 5 dos 6 erros dos modelos reais (recall de 98,25% no SGD e no SVC e de 94,74% na RF); `class_2` tem recall de 100% nos três.
- Pares de confusão: `class_1` → `class_2` (4 erros no total), `class_1` → `class_0` (1, RF), `class_0` → `class_1` (1, SVC).
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis para as próximas etapas: `plot_matrizes_confusao`, `rotulos`, `modelos`, `previsoes_treino`, `metricas_por_classe`.

## Registro da Etapa 7

- Seção "5. Teste" acrescentada (19 células, 11 de código): 5.1 SGD, 5.2 Dummy, 5.3 RF, 5.4 SVC (relatório `digits=4` com nomes das classes e `zero_division=0`, métricas macro, matriz relativa) e 5.5 Resumo. O notebook tem agora 106 células.
- O teste foi usado **pela primeira e única vez**; os modelos da Etapa 4 foram aplicados sem alteração.
- Resultados (36 amostras): Dummy 38,89% de acurácia / F1 macro 18,67% (22 erros); SGD 94,44% / 94,53% (2 erros); **RF 100% / 100% (0 erros)**; SVC 97,22% / 97,10% (1 erro).
- Erros: SGD (class_1 → class_0; class_2 → class_1); SVC (class_2 → class_1). A ordem dos modelos inverteu em relação à validação cruzada (diferenças de 1 a 2 amostras nos dois casos); não há superioridade clara.
- Correção durante a etapa: o título "Sgd Classifier" (gerado por `.title()`) foi corrigido para "SGD Classifier". O guia 06 foi ajustado: a função de 3 matrizes não é usada no teste (decisão: só a matriz relativa).
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.

## Registro da Etapa 8

- Seção "6. Comparação Final e Conclusão" acrescentada (7 células, 2 de código): tabela treino (fora da amostra) × teste com métricas macro e erros, gráfico F1/acurácia (eixo a partir de 85%, com aviso), 6.1 Interpretação e 6.2 Conclusão. O notebook tem agora 113 células.
- Nenhum modelo foi alterado; só foram reaproveitadas as previsões das Etapas 6 e 7.
- Erros: treino SGD 1, SVC 2, RF 3 (de 142); teste RF 0, SVC 1, SGD 2 (de 36). **Cada modelo errou 3 de 178 no total.** Conclusão: os três modelos são equivalentes; sem vencedor claro. 6 dos 9 erros envolvem `class_1` × `class_2`.
- Correção durante a etapa: a legenda do gráfico final cobria as barras e foi substituída por uma legenda única abaixo dos painéis.
- **Reprodutibilidade:** duas execuções em kernel limpo com **saídas idênticas** nas 53 células de código, sem erros nem avisos. Validado com `nbformat`.
- **Notebook de apresentação gerado:** `Exercise/3-Exercise.ipynb` (53 de código + 35 títulos). Executado: 0 erros, 0 avisos, 14 figuras, e saídas **idênticas** às do notebook de estudo.
- `README.md` e `CLAUDE.md` atualizados (Exercício 03 concluído). Nenhum commit.

## Decisões em vigor

Ver [`00_Plano.md` § 1.7](00_Plano.md): SGD e SVC em Pipeline com `StandardScaler`; `StratifiedKFold(5, shuffle=True, random_state=42)` único; matrizes do treino com `cross_val_predict`; `classes_[argmax]`; nomes legíveis das classes; `zero_division=0`; um único `plt.rc`; teste só na Etapa 7.

## Pendências

Nenhuma.

## Próxima etapa

Todas as etapas do Exercício 03 foram concluídas. Aguardando novas solicitações.
