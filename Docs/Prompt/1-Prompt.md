# Prompt CTRO — Exercício 01: Diagnóstico de Câncer e Otimização de Recall

## C — Contexto

Atue como especialista em Machine Learning, programação Python e ensino de Ciência de Dados.

Vou desenvolver o **Exercício 01 — Diagnóstico de Câncer e Otimização de Recall**, cujo objetivo é classificar tumores de mama em **Maligno (1)** ou **Benigno (0)**, utilizando os mesmos três modelos definidos na atividade de referência e avaliando acurácia, precisão, recall e F1-score.

O arquivo de referência está em:

`Docs/reference/1-Arquivo.ipynb`

Já preparei o ambiente virtual com `uv`, instalei as dependências iniciais e executei o `/init`. Leia as instruções existentes no projeto e confira o ambiente antes de propor alterações.

Quero construir a resolução **por etapas**, compreender as decisões e conseguir explicar os códigos e resultados ao professor.

## T — Tarefas

### 1. Primeira execução: analisar e planejar

Nesta primeira interação:

1. Leia integralmente o notebook de referência, incluindo enunciado, células Markdown, códigos, gráficos e resultados disponíveis.
2. Identifique os três modelos utilizados, a origem dos dados, o fluxo da atividade e os requisitos do exercício.
3. Inspecione a organização do projeto, o `CLAUDE.md` e o `pyproject.toml`. Aproveite o ambiente existente.
4. Desenvolva um plano completo e numerado, acompanhando a estrutura da referência.
5. Salve o plano em `Docs/Guia/00_Plano.md`, indicando, para cada etapa:
   - objetivo e conceitos envolvidos;
   - implementação prevista;
   - resultados e gráficos esperados, sem inventar valores;
   - critérios de conclusão.
6. Crie `Docs/Guia/Status.md` com o andamento e a próxima etapa.
7. Avalie se alguma skill local seria realmente útil. Crie skills em `.claude/skills/` somente quando houver uma necessidade concreta, explicando a finalidade e evitando instruções duplicadas.

**Encerre após apresentar o plano. Não implemente a resolução nem execute treinamentos nesta primeira interação. Aguarde meu pedido para iniciar a Etapa 1.**

Se o arquivo de referência estiver ausente ou não permitir identificar um requisito essencial, informe a lacuna e solicite o material necessário. Não presuma quais são os três modelos.

### 2. Organizar a resolução em etapas

Ajuste a sequência abaixo ao notebook de referência:

1. Importações e obtenção dos dados.
2. Inspeção inicial, definição do alvo e análise exploratória simples.
3. Separação dos dados de treino e teste.
4. Preparação e treinamento dos três modelos.
5. Avaliação com validação cruzada e comparação das métricas.
6. Análise do trade-off entre precisão e recall e ajuste do limiar.
7. Matrizes de confusão da avaliação no treino, usando previsões fora da amostra.
8. Avaliação final no teste, com matrizes de confusão separadas.
9. Comparação final, interpretação dos resultados e conclusão.

Preserve a ordem e o estilo da referência sempre que forem metodologicamente adequados. Explique adaptações necessárias.

### 3. Executar uma etapa por solicitação

Quando eu solicitar uma etapa:

1. Leia o plano, o status e as células já desenvolvidas.
2. Explique brevemente o objetivo da etapa.
3. Implemente **somente a etapa solicitada**.
4. Acrescente as células ao mesmo arquivo:
   `Notebooks/Exercicio_01_Cancer_Recall.ipynb`.
5. Execute as células necessárias e confira os resultados. Não avance para etapas futuras.
6. Escreva o guia correspondente em:
   `Docs/Guia/XX_Nome_da_Etapa.md`.
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
- uma sugestão de como apresentar oralmente aquela etapa.

Explique detalhadamente o raciocínio, sem repetir informações desnecessárias. Mantenha o guia sincronizado com o notebook.

## R — Restrições

### Controle de execução

- É proibido resolver toda a atividade de uma vez.
- Planejar todas as etapas não autoriza executá-las.
- Execute apenas a etapa explicitamente solicitada.
- Não antecipe códigos, resultados ou conclusões de etapas futuras.
- Não altere o notebook de referência.
- Não crie notebooks duplicados ou versões paralelas da solução.
- Se existir `Notebook/` em vez de `Notebooks/`, verifique seu conteúdo e padronize com cuidado, sem excluir arquivos.

### Simplicidade

- Use código limpo, direto e legível, adequado ao nível acadêmico da referência.
- Evite classes, abstrações, funções auxiliares e configurações sem necessidade.
- Crie funções somente quando reduzirem repetição relevante ou organizarem uma operação clara.
- Não inclua verificações de versão com `assert`.
- Não reinstale dependências nem recrie o ambiente sem necessidade.
- Não adicione frameworks, hooks ou estruturas extras sem justificativa concreta.

### Rigor metodológico

- Confirme a codificação original do alvo e garanta que **Maligno = 1** e **Benigno = 0**.
- Ajuste transformações apenas nos dados de treinamento de cada divisão; utilize `Pipeline` quando necessário para evitar vazamento na validação cruzada.
- Reserve o teste para a avaliação final. Não o utilize para escolher modelos, hiperparâmetros ou limiares.
- Use previsões fora da amostra para avaliar o treino e estudar limiares.
- Justifique a estratégia de divisão e validação; use estratificação quando adequada.
- Explique o impacto dos falsos negativos e o custo de aumentar o recall sobre a precisão e os falsos positivos.
- Não imponha uma meta arbitrária de recall nem prometa desempenho.
- Apresente matrizes de confusão separadas por modelo, com classes e eixos identificados.
- Não invente métricas ou interpretações. Se não conseguir executar algo, registre **NÃO EXECUTADO**, o motivo e o próximo passo.
- Apresente a conclusão como resultado de um exercício acadêmico, sem afirmar validade para uso clínico.
- Na etapa final, execute o notebook completo em kernel limpo para verificar a reprodução dos resultados.

## O — Saídas / Output

### Na primeira interação

Entregue:

1. Resumo da análise do arquivo de referência.
2. Plano completo em `Docs/Guia/00_Plano.md`.
3. Controle de andamento em `Docs/Guia/Status.md`.
4. Justificativa sobre a necessidade ou não de skills.
5. Indicação da primeira etapa a executar.

**Pare e aguarde meu pedido.**

### Em cada etapa posterior

Entregue:

1. Notebook atualizado apenas com o trabalho autorizado.
2. Guia didático da etapa em `Docs/Guia/`.
3. Status atualizado.
4. Resumo breve no terminal contendo:
   - o que foi feito;
   - arquivos alterados;
   - verificações e resultados;
   - eventuais pendências;
   - próxima etapa proposta.

Ao concluir cada etapa, aguarde minha solicitação para continuar.