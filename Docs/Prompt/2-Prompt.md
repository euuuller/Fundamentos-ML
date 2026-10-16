# Prompt CTRO — Exercício 02: Previsão de Sobrevivência no Titanic

## C — Contexto

Atue como especialista em Machine Learning, programação Python e ensino de Ciência de Dados.

Vou desenvolver o **Exercício 02 — Previsão de Sobrevivência no Titanic** (*Titanic - Machine Learning from Disaster*), cujo objetivo é prever se um passageiro **Sobreviveu (1)** ou **Não Sobreviveu (0)** ao naufrágio, a partir de atributos demográficos e de viagem (classe, sexo, idade, família a bordo, tarifa e porto de embarque).

Serão usados os **mesmos três modelos** da atividade de referência e do Exercício 01:

- `SGDClassifier(class_weight='balanced', random_state=42)`, com dados padronizados;
- `DummyClassifier(strategy="most_frequent", random_state=42)`, como linha de base;
- `RandomForestClassifier(class_weight='balanced', random_state=42)`.

Medidas de desempenho, conforme o enunciado: **acurácia, precisão, recall, F1-score, matriz de confusão, curva ROC e AUC**.

O arquivo de referência está em `Docs/reference/1-Arquivo.ipynb` (seção **EXERCÍCIO 2**, com o enunciado, o dicionário de atributos e a função `load_titanic_data()`).

O ambiente virtual com `uv` já está pronto, e o Exercício 01 foi concluído no mesmo repositório. Siga o mesmo modelo de trabalho: construir a resolução **por etapas**, compreender as decisões e conseguir explicar os códigos e resultados ao professor.

## T — Tarefas

### 1. Primeira execução: analisar e planejar

Já atendida. A referência foi inspecionada, as decisões foram tomadas e o plano está em:

- `Docs/Guia/2-Exercise/00_Plano.md`
- `Docs/Guia/2-Exercise/Status.md`

Decisões já definidas:

- **Modelos:** os mesmos três da referência.
- **Atributos:** `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`. Descartar `PassengerId, Name, Ticket, Cabin`, com justificativa.
- **Limiar:** analisar o *trade-off* precisão × recall e a curva ROC/AUC e escolher o limiar pelo **máximo F1**, apenas com previsões fora da amostra do treino.

### 2. Organizar a resolução em etapas

1. Importações e obtenção dos dados.
2. Inspeção inicial, definição do alvo e análise exploratória simples.
3. Separação dos dados de treino e teste.
4. Pré-processamento com `ColumnTransformer` (valores ausentes, codificação e padronização).
5. Preparação e treinamento dos três modelos.
6. Avaliação com validação cruzada e comparação das métricas (incluindo AUC).
7. *Trade-off* entre precisão e recall, curva ROC/AUC e ajuste do limiar (F1).
8. Matrizes de confusão da avaliação no treino, usando previsões fora da amostra.
9. Avaliação final no teste, com matrizes de confusão separadas e curvas ROC.
10. Comparação final, interpretação dos resultados, conclusão e notebook de apresentação.

Preserve a ordem e o estilo da referência e do Exercício 01 sempre que forem metodologicamente adequados. Explique adaptações necessárias.

### 3. Executar uma etapa por solicitação

Quando eu solicitar uma etapa:

1. Leia o plano, o status e as células já desenvolvidas.
2. Explique brevemente o objetivo da etapa.
3. Implemente **somente a etapa solicitada**.
4. Acrescente as células ao mesmo arquivo:
   `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`.
5. Execute as células necessárias e confira os resultados, inclusive os gráficos. Não avance para etapas futuras.
6. Escreva o guia correspondente em:
   `Docs/Guia/2-Exercise/XX_Nome_da_Etapa.md`.
7. Atualize o status e pare, aguardando minha próxima solicitação.

Preserve o trabalho anterior. Se encontrar um erro que comprometa a etapa atual, corrija-o e registre o motivo.

### 4. Ensinar e documentar

Mantenha o notebook enxuto, com células curtas e explicações objetivas.

No guia de cada etapa, explique como um professor:

- o objetivo e a relação com o exercício;
- o código efetivamente implementado, por blocos;
- as funções, parâmetros e variáveis relevantes;
- os conceitos e as fórmulas necessárias;
- a justificativa das escolhas;
- a interpretação dos resultados realmente obtidos;
- erros comuns e limitações;
- perguntas que o professor poderia fazer, com respostas curtas;
- uma sugestão de como apresentar oralmente aquela etapa;
- a **documentação oficial** de cada função, classe e conceito usado na etapa, com links conferidos.

Explique detalhadamente o raciocínio, sem repetir informações desnecessárias. Mantenha o guia sincronizado com o notebook.

## R — Restrições

### Controle de execução

- É proibido resolver toda a atividade de uma vez.
- Planejar todas as etapas não autoriza executá-las.
- Execute apenas a etapa explicitamente solicitada.
- Não antecipe códigos, resultados ou conclusões de etapas futuras.
- Não altere o notebook de referência nem os arquivos do Exercício 01.
- Não crie notebooks duplicados ou versões paralelas da solução.
- **Não faça commits** no repositório.

### Simplicidade

- Use código limpo, direto e legível, adequado ao nível acadêmico da referência.
- Evite classes, abstrações, funções auxiliares e configurações sem necessidade.
- Crie funções somente quando reduzirem repetição relevante ou organizarem uma operação clara.
- Não inclua verificações de versão com `assert`.
- Não reinstale dependências nem recrie o ambiente sem necessidade.
- Não adicione frameworks, hooks ou estruturas extras sem justificativa concreta.
- Não faça engenharia de atributos (título do nome, tamanho da família etc.) além dos atributos definidos; cite-a apenas como possível extensão.

### Rigor metodológico

- Baixe os dados para a pasta `datasets/` da raiz do projeto (no notebook, `../datasets`), que é ignorada pelo git.
- O `test.csv` do Kaggle **não tem a coluna `Survived`**: use **somente as linhas rotuladas** para treino e avaliação, explicando o motivo.
- Confirme a codificação do alvo e garanta que **Sobreviveu = 1** e **Não Sobreviveu = 0**.
- Faça imputação de ausentes, codificação de categóricos e padronização **dentro de `Pipeline`/`ColumnTransformer`**, ajustados apenas nos dados de treinamento de cada divisão.
- Reserve o teste para a avaliação final. Não o utilize para escolher modelos, hiperparâmetros ou limiares.
- Use previsões fora da amostra para avaliar o treino, estudar limiares e traçar as curvas.
- Justifique a estratégia de divisão e validação; use estratificação quando adequada.
- Explique o significado de falsos negativos e falsos positivos neste problema e o efeito do limiar sobre eles.
- Não imponha metas arbitrárias nem prometa desempenho.
- Apresente matrizes de confusão separadas por modelo, com classes, eixos e siglas TN/FP/FN/TP identificados.
- Interprete as relações entre atributos e sobrevivência como **associações nos dados**, sem afirmações causais ou históricas além do que os dados mostram.
- Não invente métricas ou interpretações. Se não conseguir executar algo, registre **NÃO EXECUTADO**, o motivo e o próximo passo.
- Apresente a conclusão como resultado de um exercício acadêmico.
- Na etapa final:
  - execute o notebook completo em kernel limpo para verificar a reprodução dos resultados;
  - gere o notebook de apresentação `Exercise/2-Exercise.ipynb`, com somente o código e os títulos, conferindo que as saídas são idênticas às do notebook de estudo.

## O — Saídas / Output

### Na primeira interação

Já entregues:

1. Resumo da análise do arquivo de referência.
2. Plano completo em `Docs/Guia/2-Exercise/00_Plano.md`.
3. Controle de andamento em `Docs/Guia/2-Exercise/Status.md`.
4. Justificativa sobre a necessidade ou não de skills.
5. Indicação da primeira etapa a executar.

### Em cada etapa posterior

Entregue:

1. Notebook atualizado apenas com o trabalho autorizado.
2. Guia didático da etapa em `Docs/Guia/2-Exercise/`.
3. Status atualizado.
4. Resumo breve no terminal contendo:
   - o que foi feito;
   - arquivos alterados;
   - verificações e resultados;
   - eventuais pendências;
   - próxima etapa proposta.

Ao concluir cada etapa, aguarde minha solicitação para continuar.
