# Status — Exercício 01: Diagnóstico de Câncer e Otimização de Recall

Última atualização: 2026-10-12

## Andamento

| Etapa | Descrição | Guia | Situação |
|---|---|---|---|
| 0 | Análise da referência e planejamento | [`00_Plano.md`](00_Plano.md) | ✅ Concluída |
| 1 | Importações e obtenção dos dados | [`01_Importacoes_e_Dados.md`](01_Importacoes_e_Dados.md) | ✅ Concluída |
| 2 | Inspeção inicial, definição do alvo e EDA simples | [`02_Inspecao_e_Analise_Exploratoria.md`](02_Inspecao_e_Analise_Exploratoria.md) | ✅ Concluída |
| 3 | Separação dos dados de treino e teste | [`03_Separacao_Treino_Teste.md`](03_Separacao_Treino_Teste.md) | ✅ Concluída |
| 4 | Preparação e treinamento dos três modelos | [`04_Treinamento_dos_Modelos.md`](04_Treinamento_dos_Modelos.md) | ✅ Concluída |
| 5 | Validação cruzada e comparação das métricas | [`05_Validacao_Cruzada.md`](05_Validacao_Cruzada.md) | ✅ Concluída |
| 6 | *Trade-off* precisão × recall e ajuste do limiar | [`06_Tradeoff_Precisao_Recall.md`](06_Tradeoff_Precisao_Recall.md) | ✅ Concluída |
| 7 | Matrizes de confusão no treino (fora da amostra) | [`07_Matrizes_Confusao_Treino.md`](07_Matrizes_Confusao_Treino.md) | ✅ Concluída |
| 8 | Avaliação final no teste | [`08_Avaliacao_Teste.md`](08_Avaliacao_Teste.md) | ✅ Concluída |
| 9 | Comparação final, interpretação e conclusão | [`09_Conclusao.md`](09_Conclusao.md) | ✅ Concluída |

**Exercício 01 concluído.**
- Notebook de estudo: `Notebooks/Exercicio_01_Cancer_Recall.ipynb` (129 células, 55 de código), reproduzido em kernel limpo com saídas idênticas.
- Notebook de apresentação: `Exercise/1-Exercise.ipynb` (95 células: 55 de código + 40 títulos), gerado a partir do de estudo e executado; saídas idênticas às do notebook de estudo.
- Especificação: `Docs/Prompt/1-Prompt.md`.
- Guias: `Docs/Guia/1-Exercise/`.

## Registro da Etapa 0

- Referência lida por completo; os três modelos foram identificados (SGD, Dummy `most_frequent` e Random Forest).
- Ambiente conferido: Python 3.14.7, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4, matplotlib 3.11.2, Jupyter disponível. Nada foi instalado.
- Codificação original do alvo conferida: `target_names = ['malignant', 'benign']` (0 = maligno). Será invertida na Etapa 2.
- Pasta `Notebook/` (vazia) renomeada para `Notebooks/`, conforme a especificação. Nenhum arquivo foi excluído.
- Nenhum notebook de solução criado e nenhum treinamento executado.
- Skills: nenhuma criada (ver justificativa abaixo).

## Registro da Etapa 1

- Notebook `Notebooks/Exercicio_01_Cancer_Recall.ipynb` criado com as seções "1. Observar o Panorama Geral" e "2. Obtenção dos Dados" (início): 9 células, 4 de código.
- Executado em kernel limpo (`nbconvert --execute`) sem erros; formato validado com `nbformat.validate`.
- Resultados: `X.shape = (569, 30)`, `y_original.shape = (569,)`, 30 atributos listados.
- Ajuste em relação ao plano: só `matplotlib` e `load_breast_cancer` foram importados; `numpy` e `pandas` entram na Etapa 2, onde são usados (estilo da referência e sem antecipar etapas).
- O alvo ainda está na codificação original (`y_original`); a conferência e a conversão ficam para a Etapa 2.

## Registro da Etapa 2

- Seções "2.1. Definição do Alvo" e "2.2. Descrição dos Dados" acrescentadas (13 células, 8 de código); o notebook tem agora 22 células.
- Codificação original comprovada no notebook: `['malignant' 'benign']`, ou seja, 0 = maligno (212) e 1 = benigno (357). Alvo convertido para `y = 1 - y_original` (Maligno = 1) e conferido por tabela.
- Nulos: 0 em `X` e em `y`. Distribuição: Benigno 357 (62,74%) × Maligno 212 (37,26%).
- Escalas: `mean area` de 143,5 a 2501; `mean smoothness` de 0,053 a 0,163 (justifica padronizar para o SGD).
- Notebook executado em kernel limpo sem erros e validado com `nbformat`.
- Variáveis criadas para as próximas etapas: `y` (alvo oficial), `df`, `nomes_classes`.

## Registro da Etapa 3

- Seção "2.3. Separação dos Dados de Treino e Teste" acrescentada (5 células, 2 de código); o notebook tem agora 27 células.
- `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`: treino (455, 30) com 170 malignos e teste (114, 30) com 42 malignos.
- Proporção de malignos: 37,26% completo, 37,36% treino e 36,84% teste. A estratificação foi confirmada.
- Correção durante a etapa: a célula de proporções foi gerada com uma quebra de linha real dentro de um `print("\n...")`, o que causou `SyntaxError`. Ela foi corrigida com `print()` seguido de `print("Proporção ...")`, e o notebook foi reexecutado sem erros.
- Notebook executado em kernel limpo e validado com `nbformat`.
- `X_test` e `y_test` estão **reservados para a Etapa 8**.

## Registro da Etapa 4

- Seção "3. Treinamento dos Modelos de Classificação Binária" acrescentada (17 células, 8 de código); o notebook tem agora 44 células.
- Modelos treinados só com `X_train`/`y_train`: `sgd_clf = make_pipeline(StandardScaler(), SGDClassifier(class_weight='balanced', random_state=42))`, `dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)` e `forest_clf = RandomForestClassifier(class_weight='balanced', random_state=42)`.
- Pesos `balanced`: Benigno 0,798 e Maligno 1,338. Scaler (treino) para `mean area`: média 659,578 e desvio 360,022.
- Exemplo (1º maligno do treino, índice 0): SGD → Maligno, Dummy → Benigno, RF → Maligno. É só uma verificação de funcionamento, não uma métrica.
- Correções durante a etapa:
  1. A fórmula LaTeX da célula de `class_weight` foi gravada com caracteres de controle (`\b` e `\f` interpretados como escapes ao gerar a célula pelo shell). Foi reescrita com strings *raw* a partir de um script em arquivo, e confirmou-se que nenhuma célula tem caracteres de controle.
  2. O exemplo de previsão (índice 16, benigno, igual à referência) foi trocado pelo primeiro tumor maligno do treino, porque com um benigno os três modelos acertariam e a comparação não seria informativa.
- Notebook executado em kernel limpo e validado com `nbformat`.

## Registro da Etapa 5

- Seções "4. Avaliação de Performance" (definições e fórmulas) e "4.1 Validação Cruzada" acrescentadas (25 células, 7 de código); o notebook tem agora 69 células.
- `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`; métricas: accuracy, precision e f1 (via `make_scorer(..., zero_division=0)`) e recall. Sem avisos na execução.
- Resultados (média ± desvio, 5 dobras, limiar padrão):
  - Dummy: acurácia 62,64%; precisão, recall e F1 = 0%.
  - SGD: acurácia 96,92% ± 0,82%; precisão 96,08% ± 3,35%; recall 95,88% ± 3,00%; F1 95,89% ± 1,06%.
  - RF: acurácia 95,82% ± 1,62%; precisão 94,18% ± 2,56%; recall 94,71% ± 2,20%; F1 94,43% ± 2,14%.
- FN somados nas dobras: SGD 7/170 e RF 9/170. A diferença entre SGD e RF é menor que a variação entre dobras, então não há superioridade clara.
- Variáveis criadas para as próximas etapas: `cv`, `metricas`, `nomes_metricas`, `resultados_sgd`, `resultados_dummy`, `resultados_forest`, `df_comparacao`.
- Notebook executado em kernel limpo (resultados idênticos em duas execuções) e validado com `nbformat`.

## Registro da Etapa 6

- Subseção "Trade-off da Precisão e Recall" acrescentada à seção 4 (13 células, 5 de código); o notebook tem agora 82 células.
- Pontuações fora da amostra com `cross_val_predict` (mesmo `cv`): `decision_function` (SGD) e `predict_proba[:, 1]` (RF).
- Limiares escolhidos pelo máximo F2, só com o treino: **`limiar_sgd = -2.9289`** (padrão 0) e **`limiar_forest = 0.45`** (padrão 0,5). Não houve empate no F2 máximo.
- Resultado no treino (fora da amostra): SGD com FN 7 → 4 e FP 7 → 13 (recall 95,88% → 97,65%; precisão 95,88% → 92,74%); RF com FN 9 → 6 e FP 10 → 13 (recall 94,71% → 96,47%; precisão 94,15% → 92,66%).
- Ajustes durante a etapa: (1) a tabela final deixou de usar `Styler` (sem saída em texto) e passou a ser um DataFrame formatado; (2) os gráficos ganharam zoom (painel ampliado na curva P×R e eixo X do SGD em [−40, 40]), porque as diferenças relevantes ficavam invisíveis.
- Variáveis para as próximas etapas: `y_scores_sgd`, `y_scores_forest`, `limiar_sgd`, `limiar_forest`, `df_limiares`.
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.

## Registro da Etapa 7

- Seção "4.2 Matriz de Confusão" acrescentada (16 células, 7 de código); o notebook tem agora 98 células.
- Função `plot_matriz_confusao(y_real, y_pred, titulo)` criada (rótulos Benigno (0)/Maligno (1), eixos "Classe real"/"Classe prevista"); será reutilizada na Etapa 8.
- 5 matrizes separadas, no treino e fora da amostra (mesmo `cv`): Dummy, SGD padrão, SGD F2, RF padrão e RF F2.
- Contagens (TN, FP, FN, TP): Dummy 285/0/170/0; SGD padrão 278/7/7/163; SGD F2 272/13/4/166; RF padrão 275/10/9/161; RF F2 272/13/6/164. São consistentes com as Etapas 5 e 6.
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- **Ajuste a pedido do usuário:** `plot_matriz_confusao` passou a escrever as siglas TN/FP/FN/TP em negrito acima de cada valor, na mesma cor do número. As 5 matrizes foram regeneradas e as contagens não mudaram. Guia 07 atualizado (código, explicação e links). As matrizes do teste (Etapa 8) já sairão nesse formato.

## Registro da Etapa 8

- Seção "5. Teste" acrescentada (24 células, 13 de código); o notebook tem agora 122 células.
- O teste foi usado **pela primeira e única vez**. Os modelos da Etapa 4 e os limiares da Etapa 6 foram aplicados sem alteração.
- Função `metricas_teste(y_pred)` criada (acurácia, precisão, recall, F1, F2). Há 5 matrizes de confusão separadas, com siglas.
- Resultados no teste (72 benignos, 42 malignos):
  - Dummy: acurácia 63,16%, recall 0%, FN 42.
  - SGD padrão: acurácia 92,11%, precisão 86,67%, recall 92,86%, F1 89,66% (FN 3, FP 6).
  - SGD F2: acurácia 92,98%, precisão 86,96%, **recall 95,24%**, F1 90,91% (FN 2, FP 6).
  - RF padrão: **acurácia 97,37%, precisão 100%**, recall 92,86%, F1 96,30% (FN 3, FP 0).
  - RF F2: acurácia 96,49%, precisão 97,50%, recall 92,86%, F1 95,12% (FN 3, FP 1).
- O ajuste F2 ganhou 1 TP sem FP adicional no SGD e não trouxe ganho na RF (+1 FP). As diferenças de recall são de 1 tumor.
- Análise auxiliar das pontuações (fora do notebook, só explicativa) registrada no guia 08.
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.

## Registro da Etapa 9

- Seção "6. Comparação Final e Conclusão" acrescentada (7 células, 2 de código): tabela treino (fora da amostra) × teste, gráfico de recall e precisão, 6.1 Interpretação e 6.2 Conclusão. O notebook tem agora 129 células.
- Nenhum modelo, parâmetro ou limiar foi alterado; só foram reaproveitadas previsões já calculadas.
- Ajustes no gráfico durante a etapa: a legenda foi movida para não cobrir as barras; o eixo Y começa em 80%, com aviso explícito no rótulo, e as marcas vão até 100%.
- Conclusão: SGD - F2 teve o maior recall (treino 97,65%, teste 95,24%); RF padrão teve a maior acurácia, precisão e F1 no teste (97,37%, 100%, 96,30%; 0 FP). A diferença de recall no teste é de 1 tumor, sem superioridade estabelecida. Há aviso de que não há validade clínica.
- **Reprodutibilidade:** o notebook completo foi reexecutado em kernel limpo (`nbconvert --execute`). As 55 células de código rodaram em sequência, sem erros nem avisos, e as saídas de texto são **idênticas** às da execução anterior. Validado com `nbformat`.

## Reorganização do projeto (a pedido do usuário, após a Etapa 9)

- Os guias do Exercício 01 foram movidos de `Docs/Guia/` para **`Docs/Guia/1-Exercise/`**, preparando o repositório para novas atividades. Os links relativos do `00_Plano.md` foram corrigidos.
- Foi criado o **notebook de apresentação**, com as 55 células de código do notebook de estudo, na mesma ordem. Foi executado em kernel limpo: 0 erros, 0 avisos, 18 figuras, e as saídas de texto são **idênticas** às do notebook de estudo célula a célula.
- **Segundo ajuste (a pedido do usuário):** (1) o arquivo foi renomeado de `Exercise/1- Exercise.ipynb` para **`Exercise/1-Exercise.ipynb`**; (2) foram incluídos **somente os títulos** do notebook de estudo (40 títulos: cabeçalhos `#` e subtítulos em negrito que antecedem código; sem textos explicativos nem "Observações"); (3) a especificação foi renomeada de `Docs/Prompt/Prompt.md` para **`Docs/Prompt/1-Prompt.md`**, preparando o `2-Prompt.md` do Exercício 02 (Titanic). O notebook foi regerado e reexecutado, e as saídas continuam idênticas às do notebook de estudo.
- O notebook de estudo (`Notebooks/Exercicio_01_Cancer_Recall.ipynb`) não foi alterado.
- `CLAUDE.md` e `README.md` foram atualizados com a estrutura por exercício.
- O conteúdo da especificação (hoje `Docs/Prompt/1-Prompt.md`) **não foi alterado**. Os caminhos citados nele (`Docs/Guia/...`) são os da época da especificação; os atuais estão no `00_Plano.md`.

## Ajuste nos guias (a pedido do usuário)

- Nova seção **"10. Documentação oficial"** acrescentada aos guias 01 e 02, com links conferidos (HTTP 200 e âncoras existentes).
- O modelo de guia no `00_Plano.md` (§ 2) agora inclui essa seção; todas as etapas seguintes a trarão.

## Decisões em vigor

Ver detalhes em [`00_Plano.md` § 1.5](00_Plano.md):
alvo `y = 1 - cancer.target`; SGD em `Pipeline` com `StandardScaler`; `StratifiedKFold(5, shuffle=True, random_state=42)` único; limiar pelo máximo de F2 com previsões fora da amostra do treino; teste usado só na Etapa 8.

## Skills

Nenhuma skill local foi criada. O fluxo por etapa e o modelo do guia estão documentados uma única vez no `00_Plano.md` (§ 2), com uma referência no `CLAUDE.md`. Uma skill repetiria essas instruções sem ganho concreto para um único exercício de 9 etapas. A decisão será reavaliada se o mesmo fluxo for reaproveitado no Exercício 2.

## Pendências

Nenhuma.

## Próxima etapa

Todas as etapas do Exercício 01 foram concluídas. Não há próxima etapa planejada; aguardando novas solicitações (por exemplo, revisão final ou o Exercício 2).
