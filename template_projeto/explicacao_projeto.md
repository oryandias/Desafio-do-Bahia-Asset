# Explicação Técnica do Projeto: Divisão Justa de Execuções entre Fundos de Investimento via Algoritmo Genético

**Candidato:** Ryan Dias  
**Instituição de Origem:** Universidade Federal do Rio de Janeiro (UFRJ)  
**Desafio Técnico:** Algoritmo Genético para Divisão de Execuções entre Fundos  

---

## 1. Resumo Executivo e Contexto de Negócio

No ecossistema operacional de uma gestora de recursos (*Asset Management*), ordens institucionais de compra e venda de grande magnitude são emitidas em bloco na mesa de renda variável para atender simultaneamente a múltiplos fundos de investimento com cotas, mandatos e passivos distintos.

Ao longo do pregão diário da B3, uma ordem institucional consolidada de **100.000 ações** é executada em sucessivas tranches (lotes de liquidação) com preços unitários que oscilam conforme a volatilidade do mercado. Para a base sob análise, o Preço Médio Ponderado Global ($PU_{\text{global}}$) de fechamento foi estabelecido no valor fiduciário de referência de **R$ 10,48** em todos os 500 cenários simulados.

```
+-----------------------------------------------------------------------------------+
|                            ORDEM INSTITUCIONAL CONSOLIDADA                        |
|                                     100.000 Ações                                 |
|                         PU Médio Global da Ordem = R$ 10,48                       |
+-----------------------------------------------------------------------------------+
                                         │
                   Execuções Fracionadas ao Longo do Pregão (N Lotes)
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                         ALOCAÇÃO REGULATÓRIA ENTRE FUNDOS                         |
|  • Fundo 1: 47.000 ações (47,0%)             • Fundo 5:  2.700 ações ( 2,7%)      |
|  • Fundo 2: 23.000 ações (23,0%)             • Fundo 6:    400 ações ( 0,4%)      |
|  • Fundo 3: 21.600 ações (21,6%)             • Fundo 7:    200 ações ( 0,2%)      |
|  • Fundo 4:  5.100 ações ( 5,1%)             TOTAL:    100.000 ações              |
+-----------------------------------------------------------------------------------+
```

### O Desafio Operacional e Regulatório
A missão do sistema de *Order Management System (OMS)* e do Backoffice consiste em alocar cada lote de execução entre os 7 fundos da casa de modo a cumprir simultaneamente dois pilares regulatórios indispensáveis (instruções CVM 555 / 175 e Código ANBIMA de Administração de Recursos):

1. **Restrições Rígidas de Integridade (Hard Constraints):**
   - **Exatidão das Cotas Contratadas:** Cada fundo $f$ deve receber estritamente sua cota $A_f$ ($\sum_{j=1}^N M_{j, f} = A_f$).
   - **Quantidades Estritamente Inteiras:** É vedada a liquidação de frações decimais de ações na custódia dos fundos ($M_{j, f} \in \mathbb{N}$).
   - **Conservação de Massa por Lote:** Cada lote $j$ de tamanho $Q_j$ deve ser integralmente absorvido pelos fundos sem criação ou supressão de ações ($\sum_{f=1}^7 M_{j, f} = Q_j$).
2. **Equidade Fiduciária (*Fairness*):**
   - O Preço Médio Unitário ($PU_f$) de cada fundo deve coincidir ao máximo com o benchmark global da ordem ($PU_{\text{global}} = \text{R\$} 10{,}48$). Nenhuma carteira pode ser beneficiada com os lotes mais baratos do pregão em detrimento de fundos menores que recebam apenas lotes inflacionados.

---

## 2. Modelagem Matemática e Função de Aptidão (Fitness)

### 2.1. Formulação do Preço Médio por Fundo
Dada uma matriz de decisão $M \in \mathbb{N}^{N \times 7}$, onde $M_{j, f}$ indica o número de ações da execução $j$ destinadas ao fundo $f$, o Preço Médio Unitário ponderado pelo financeiro é expresso por:

$$PU_f = \frac{\sum_{j=1}^{N} M_{j, f} \cdot P_j}{A_f}$$

onde $P_j$ é o preço unitário do lote $j$ e $A_f$ é a cota total do fundo $f$.

### 2.2. A Decisão Fiduciária: Erro Quadrático Médio (MSE) vs. Erro Absoluto Médio (MAE)
A métrica de otimização escolhida é o **Erro Quadrático Médio (MSE)**:

$$\text{MSE} = \frac{1}{7} \sum_{f=1}^{7} (PU_f - PU_{\text{global}})^2$$

**Justificativa Técnica para a Escolha do MSE:**
* **Hiper-sensibilidade das Carteiras Menores:** O Fundo 1 detém 47.000 ações (47% do volume), enquanto o Fundo 7 detém apenas 200 ações (0,2%). Devido ao denominador reduzido ($A_7 = 200$), qualquer lote com desvio de preço causa uma distorção percentual acentuada no PU do Fundo 7.
* **Penalização Quadrática de Outliers:** A função linear (MAE, $\frac{1}{7}\sum |PU_f - PU_{\text{global}}|$) permite que o otimizador sacrifique o Fundo 7 (gerando, por exemplo, um desvio de R$ 0,0050) em troca de um ganho marginal insignificante nos Fundos 1 e 2. O **MSE penaliza os desvios ao quadrado**, tornando soluções com discrepâncias pontuais inaceitáveis e forçando a convergência coordenada de todos os fundos.

### 2.3. Formulação da Aptidão (Fitness)
Para viabilizar a seleção evolutiva proporcional via torneio e manter o espaço numérico contínuo no intervalo normalizado $(0, 1]$:

$$\text{Fitness} = \frac{1}{1 + \text{MSE}}$$

* Para a solução ideal ($\text{MSE} = 0$), o $\text{Fitness} = 1{,}000000$ (100%).
* Não há assíntotas ou indeterminações numéricas, garantindo estabilidade ao longo do processo evolutivo.

### 2.4. Validação Empírica: Benchmark MSE vs. MAE
O script [`benchmark_grafico.py`] realiza a comparação estatística entre os dois critérios sob condições de contorno idênticas.

![Validação Empírica MSE vs MAE](comparativo_mse_vs_mae.png)

**Conclusões Técnicas da Validação Empírica:**
1. **Proteção Fiduciária aos Fundos Menores (Painel A):** Sob otimização com MAE, o Fundo 7 registra desvio de PU superior a **R$ 0,0022**. Sob MSE, a penalidade quadrática reduz o desvio do Fundo 7 para menos da metade (**< R$ 0,0010**), equalizando o tratamento dispensado a cotistas institucionais e pequenos.
2. **Convergência Acelerada (Painel B):** A superfície quadrática do MSE possui gradiente suave e estritamente convexo, eliminando platôs de aptidão e alcançando o patamar de estabilização em até 20 gerações.
3. **Análise de Dispersão e Controle de Outliers (Painel C):** Em teste amostral cobrindo 350 pontos de dados (50 cenários $\times$ 7 fundos), ambas as funções demonstram alta precisão agregada, com desvios médios e desvios padrão idênticos até a 4ª casa decimal (na casa de R$ 0,0004 a R$ 0,0005, ou meio milicêntimo de real). A distinção fiduciária crítica reside nos casos extremos (outliers): enquanto a penalidade linear do MAE tolera que carteiras menores acumulem desvios pontuais maiores (pois o impacto na soma global é diluído), o MSE eleva a punição ao quadrado ($e^2$), eliminando distorções atípicas e assegurando equidade rigorosa perante a auditoria.

---

## 3. Desafios Enfrentados e Decisões de Arquitetura

### Desafio 1: O Espaço de Busca Restrito e a Decomposição Pro-Rata

#### 1. A Inviabilidade de Abordagens Aleatórias Ingênuas
Em cenários com até $1.000$ execuções e $7$ fundos, o espaço de busca consiste em uma matriz inteira $M \in \mathbb{N}^{N \times 7}$ com até **$7.000$ variáveis de decisão**, sujeita a restrições bilaterais de igualdade estrita:
* $\sum_{f=1}^7 M_{j, f} = Q_j, \quad \forall j \in \{1, \dots, N\}$ (Conservação das linhas / lotes de mercado)
* $\sum_{j=1}^N M_{j, f} = A_f, \quad \forall f \in \{1, \dots, 7\}$ (Conservação das colunas / cotas dos fundos)

Um algoritmo genético que tente sortear valores inteiros aleatórios ou aplicar mutações cegas possui **probabilidade matematicamente nula ($P \approx 0$)** de gerar matrizes viáveis. Utilizar funções de penalidade para soluções inviáveis (*death penalty*) faria a população ficar estagnada em regiões não conformes, tornando o algoritmo inoperante para o Backoffice.

---

#### 2. Solução Arquitetural: Decomposição em Piso Inteiro Garantido e Ações Residuais
Para resolver de forma determinística a viabilidade e acelerar a busca, a matriz de alocação $M$ é decomposta analiticamente em duas partes desacopladas:

$$M_{j, f} = \text{Base}_{j, f} + X_{j, f}$$

##### A) O Piso Inteiro Garantido ($\text{Base}_{j, f}$)
A cota teórica contínua que cada fundo receberia do lote $j$ pela regra de proporcionalidade pura (*pro-rata*) é:

$$\text{Cota Teórica}_{j, f} = Q_j \times \frac{A_f}{100.000}$$

Se a liquidação em bolsa aceitasse cotas fracionárias, essa divisão entregaria exatamente $PU_f = \text{R\$} 10{,}48$ para todas as carteiras em todas as execuções, com **erro zero absoluto**.

Como a regulação do mercado acionário exige quantidades inteiras, extraímos a maior parcela inteira garantida por meio da função piso (*floor*):

$$\text{Base}_{j, f} = \left\lfloor Q_j \times \frac{A_f}{100.000} \right\rfloor$$

```
+──────────────────────────────────────────────────────────────────────────────────────────+
| EXECUÇÃO j: 1.000 ações  │  Fundo 1 (47%)  │  Fundo 2 (23%)  │  ...  │  Fundo 7 (0,2%)   |
+──────────────────────────┼─────────────────┼─────────────────┼───────┼───────────────────+
| Cota Teórica Contínua    │   470,0 ações   │   230,0 ações   │  ...  │     2,0 ações     |
| Piso Inteiro Garantido   │   470   ações   │   230   ações   │  ...  │     2   ações     |
| Resíduo Fracionário      │     0,0 ações   │     0,0 ações   │  ...  │     0,0 ações     |
+──────────────────────────┼─────────────────┼─────────────────┼───────┼───────────────────+
| EXECUÇÃO k:    26 ações  │  Fundo 1 (47%)  │  Fundo 2 (23%)  │  ...  │  Fundo 7 (0,2%)   |
+──────────────────────────┼─────────────────┼─────────────────┼───────┼───────────────────+
| Cota Teórica Contínua    │   12,22 ações   │    5,98 ações   │  ...  │    0,052 ações    |
| Piso Inteiro Garantido   │   12    ações   │    5    ações   │  ...  │    0     ações    |
| Resíduo Fracionário      │    0,22 ações   │    0,98 ações   │  ...  │    0,052 ações    |
+──────────────────────────┴─────────────────┴─────────────────┴───────┴───────────────────+
```

* **Comportamento nos Fundos Maiores (Fundos 1, 2 e 3):** Como detêm 47%, 23% e 21,6% da ordem, a função piso aloca entre **98% e 99%** de suas cotas diretamente na base inteira. Dessa forma, a base de preço médio dessas carteiras já nasce colada em R$ 10,48.
* **O Fenômeno do Truncamento nos Fundos Menores (Fundos 6 e 7):** O Fundo 7 detém cota de apenas 200 ações ($0{,}2\%$). Em execuções com volume menor que 500 ações (frequentes no mercado fracionário ou em fatiamento de lotes), sua cota teórica é menor que $1{,}0$ (ex: $26 \times 0{,}002 = 0{,}052$). O piso resulta em $\lfloor 0{,}052 \rfloor = 0$ ações.
  * *Exemplo Real (Cenário 1):* Das 554 execuções do Cenário 1, o Fundo 7 recebe apenas 1 ação na base (em um lote grande executado na máxima de R$ 12,32). Seu preço médio preliminar provisório fica distorcido em **R$ 12,32** (+R$ 1,84 acima do benchmark), restando **199 cotas pendentes**.

---

##### B) As Ações Residuais ($X_{j, f}$) e a Missão do Algoritmo Genético
O truncamento das partes decimais gera dois balanços complementares que devem ser reconciliados:
1. **Resíduo por Execução ($R_j$):** Ações remanescentes no lote $j$ após a aplicação do piso inteiro:
   $$R_j = Q_j - \sum_{f=1}^{7} \text{Base}_{j, f} \quad \implies \quad R_j \in \{0, 1, 2, 3, 4, 5, 6\}$$
2. **Saldo Pendente por Fundo ($\Delta_f$):** Ações necessárias para completar a cota contratada de cada fundo:
   $$\Delta_f = A_f - \sum_{j=1}^{N} \text{Base}_{j, f}$$

Pelo princípio da conservação contábil:

$$\sum_{j=1}^{N} R_j = \sum_{f=1}^{7} \Delta_f$$

A soma total das ações residuais espalhadas pelas execuções coincide exatamente com o total de ações pendentes nos fundos.

**O Papel Cirúrgico do Algoritmo Genético:**
Em vez de tentar alocar 100.000 ações às cegas, o AG atua **exclusivamente na alocação da matriz residual $X_{j, f}$**:
* Ele distribui as ações residuais dos lotes negociados abaixo de R$ 10,48 (por exemplo, a R$ 8,80 ou R$ 9,20) para o Fundo 7, neutralizando a execução pontual de R$ 12,32 e puxando a média ponderada do fundo para R$ 10,48.
* Toda solução gerada pela população é **100% viável por construção**: respeita as somas das linhas ($R_j$) e as somas das colunas ($\Delta_f$), eliminando indivíduos inválidos e permitindo que as iterações evolutivas foquem integralmente na minimização do MSE.

---

### Desafio 2: Operadores Genéticos Factíveis por Construção

Para garantir que a busca evolutiva não destrua a conformidade das matrizes geradas, os operadores genéticos foram concebidos com mecanismos de preservação:

1. **Crossover Horizontal com Reparo Linear $O(N)$:**
   O operador realiza um corte horizontal em um ponto de execução $k \in \{1, \dots, N-1\}$. As linhas até $k$ vêm do Pai 1 e as linhas restantes vêm do Pai 2. Como cada linha isolada preserva a conservação do lote $j$, as somas de linhas mantêm-se exatas. Se houver desbalanço nas somas das colunas dos fundos, um algoritmo de reparo linear em tempo $O(N)$ transfere cotas excedentes de fundos superavitários para os deficitários, restabelecendo a viabilidade com custo computacional desprezível.

2. **Mutação por Ciclos Fechados (Swaps Alternantes $2 \times 2$):**
   Para explorar a vizinhança sem violar nenhuma restrição de linha ou coluna, a mutação opera através de trocas alternadas:
   - Selecionam-se duas execuções distintas com preços diferentes ($j_1, j_2$) e dois fundos ($f_1, f_2$) onde haja cotas alocadas.
   - Aplica-se a translação em circuito fechado:
     $$\Delta M_{j_1, f_1} = -\delta, \quad \Delta M_{j_1, f_2} = +\delta, \quad \Delta M_{j_2, f_2} = -\delta, \quad \Delta M_{j_2, f_1} = +\delta$$
   - Como a variação líquida em cada linha é $(-\delta + \delta) = 0$ e em cada coluna é $(-\delta + \delta) = 0$, **nenhuma soma é alterada**. O preço médio dos fundos é refinado sem necessidade de reparo adicional.

---

## 4. Estrutura do Projeto, Engenharia de Software e Clean Code

### 4.1. Arquitetura Modular e Organização de Pastas
O código funcional foi encapsulado em um pacote dedicado (`projeto_bahia_asset`), preservando a raiz do repositório limpa e mantendo na pasta de entrega (`template_projeto/`) apenas os executáveis, relatórios e artefatos de entrega:

```
BahiaAsset_Algo_Genetico/
│
├── alocacao_ordem.csv                     # Dados de entrada da ordem
├── massa_execucoes_500_cenarios.csv       # Execuções dos 500 cenários simulados
├── preco_medio_por_cenario.csv            # Preços médios de referência
├── README.md                              # Documentação original do desafio (12 linhas)
│
└── template_projeto/                      # Diretório de entrega e execução
    ├── divisao_execucoes.py               # Ponto de entrada oficial (CLI do desafio)
    ├── benchmark_grafico.py               # Script de benchmarking empírico (MSE vs MAE)
    ├── comparativo_mse_vs_mae.png         # Gráfico comprobatório gerado
    ├── resultado_divisao.xlsx             # Planilha com 250.000 alocações consolidadas
    ├── explicacao_projeto.md              # Documentação técnica e arquitetural
    ├── requirements.txt                   # Dependências do projeto
    └── projeto_bahia_asset/               # Pacote funcional das regras de negócio
        ├── __init__.py                    # Interface pública do pacote
        ├── config.py                      # Parameter Object com hiperparâmetros
        ├── dados.py                       # Camada de I/O, resolução de caminhos e Excel
        ├── algoritmo_genetico.py          # Núcleo evolutivo (classe AlgoritmoGenetico)
        └── main.py                        # Orquestrador da execução dos cenários
```

---

### 4.2. Padrões de Projeto e Práticas de Código Limpo Aplicadas

1. **Princípio da Responsabilidade Única (*Single Responsibility Principle* - SRP):**
   - [`config.py`]: Responsável exclusivamente por encapsular os hiperparâmetros evolutivos.
   - [`dados.py`]: Responsável pela leitura, validação estrutural e exportação dos dados.
   - [`algoritmo_genetico.py`]: Isola o modelo matemático e os operadores de otimização evolutiva.
   - [`main.py`]: Orquestra o loop de cenários e o cálculo das métricas de monitoramento.

2. **Parameter Object Pattern:**
   Os hiperparâmetros evolutivos são concentrados na classe imutável `@dataclass ConfiguracaoAG`, evitando a dispersão de variáveis globais e facilitando eventuais baterias de calibração (*grid search*):
   ```python
   @dataclass
   class ConfiguracaoAG:
       tamanho_populacao: int = 25
       geracoes: int = 35
       taxa_crossover: float = 0.70
       taxa_mutacao: float = 0.50
       num_elite: int = 2
       tamanho_torneio: int = 3
   ```

3. **Injeção de Dependências e Construtores Limpos:**
   A classe `AlgoritmoGenetico` recebe as matrizes do cenário uma única vez em seu método construtor `__init__(cotas_fundos, qtd_execucoes, precos, configuracao)`. Isso elimina a necessidade de repassar múltiplos parâmetros repetitivos para cada função interna (`avaliar()`, `crossover()`, `mutar()`), tornando a interface dos métodos simples e legível.

---

## 5. Resultados Operacionais, SLA e Conformidade de Backoffice

### 5.1. Métricas de Performance e SLA Operacional
A solução foi executada sobre os 500 cenários disponibilizados (totalizando mais de **250.000 linhas de execução**):

| Métrica de Aferição | Resultado Obtido | Meta / Padrão de Mercado |
| :--- | :--- | :--- |
| **Tempo Total de Processamento** | **84,3 segundos (~1,4 min)** | < 15 minutos (fechamento D+0) |
| **Tempo Médio por Cenário** | **~0,17 segundos** | < 1,00 segundo |
| **Erro Quadrático Médio (MSE Médio)** | **$3{,}94 \times 10^{-7}$** | < $10^{-4}$ |
| **Desvio Máximo Médio por Fundo** | **R$ 0,0018** (< 2 décimos de centavo) | < R$ 0,01 (1 centavo) |
| **Conformidade de Cotas (Hard Constraint)** | **100% de exatidão** (0 falhas) | 100% regulatório |
| **Conservação de Massa por Lote** | **100% de exatidão** (0 resíduos) | 100% contábil |

### 5.2. Estrutura da Planilha Gerada (`resultado_divisao.xlsx`)
O arquivo final gerado pelo script contém duas abas detalhadas:
* **Aba 1 (`Divisao_Execucoes`):** 250.000 linhas com colunas para `cenario_id`, `execucao_id`, `quantidade`, `preco`, `tipo_distribuicao`, e as 7 colunas dedicadas aos fundos (`Fundo 1` até `Fundo 7`), prontas para importação direta no sistema de mensageria da custódia.
* **Aba 2 (`Metricas_Cenarios`):** Sumário analítico por cenário, listando `mse`, `fitness`, `pu_global`, `max_desvio_fundo` e o tempo de execução em segundos para fins de governança e auditoria do Backoffice.

---

## 6. Considerações Finais e Próximos Passos para Produção

O algoritmo desenvolvido atende integralmente a todos os requisitos do desafio técnico e opera com eficiência computacional compatível com os horários de corte de fechamento da B3 (*cut-off* D+0).

### Recomendações de Evolução Contínua para Ambiente de Produção:
1. **Otimização de Custos de Liquidação (Minimização de Fragmentação de Boletas):**
   Incorporar um termo de regularização L1 ou contagem de boletas no fitness para agrupar as alocações em blocos maiores quando possível, reduzindo emolumentos e taxas de liquidação por bilhete na câmara B3.
2. **Execução Concorrente Multiprocessada:**
   Como os cenários são estocasticamente independentes, a função `executar()` pode facilmente ser adaptada com `concurrent.futures.ProcessPoolExecutor` para distribuir os cenários entre os núcleos do servidor, reduzindo o tempo total de processamento de 84 segundos para **menos de 20 segundos**.
3. **Módulo de Reconciliação Automática de D+0:**
   Implementação de rotina de batimento em tempo real que envie alertas automáticos caso algum lote apresente desvio fiduciário superior à tolerância contratada da gestora.
