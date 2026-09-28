# Explicação Técnica do Projeto: Divisão Justa de Execuções entre Fundos de Investimento via Algoritmo Genético

**Candidato:** Ryan Dias  
**Instituição de Origem:** UFRJ  
**Desafio Técnico:** Algoritmo Genético para Divisão de Execuções entre Fundos  
**Instituição Avaliadora:** Bahia Asset Management  

---

## 1. Resumo Executivo e Contexto de Negócio

No ambiente de gestão de recursos de uma *Asset Management*, ordens institucionais de grande porte em renda variável são emitidas para atender simultaneamente a múltiplos fundos de investimento com mandatos, passivos e cotas distintos. 

Ao longo do pregão, uma ordem total de compra de **100.000 ações** é executada em diversas tranches (lotes/execuções) a preços unitários oscilantes. O Preço Médio Ponderado Global dessa ordem foi estabelecido em **R$ 10,48** para todos os 500 cenários analisados.

A missão deste projeto consiste em desenvolver um **Algoritmo Genético (AG)** em Python robusto, genérico e computacionalmente eficiente, capaz de alocar cada uma das $N$ execuções entre os 7 fundos da casa respeitando:
1. **Restrições Rígidas de Integridade (Hard Constraints):**
   - Respeito estrito às alocações totais contratadas por cada fundo (Fundo 1: 47.000 cotas; Fundo 2: 23.000; Fundo 3: 21.600; Fundo 4: 5.100; Fundo 5: 2.700; Fundo 6: 400; Fundo 7: 200).
   - Alocações inteiras (proibição regulatória de cotas/ações fracionárias).
   - Conservação de massa por execução (a soma das ações distribuídas aos fundos em cada lote deve ser idêntica à quantidade executada daquele lote).
2. **Equidade e Fairness Regulatória (CVM / ANBIMA):**
   - Garantir que o Preço Médio Unitário ($PU_f$) de cada fundo fique o mais próximo possível do Preço Médio da Ordem ($PU_{\text{global}} = \text{R\$} 10{,}48$), impedindo privilégio ou prejuízo injusto entre carteiras.

---

## 2. Modelagem Matemática e Função de Aptidão (Fitness)

### 2.1. Cálculo do Preço Médio por Fundo
Para uma matriz de alocação $M \in \mathbb{N}^{N \times 7}$, onde $M_{j, f}$ representa o número de ações da execução $j$ destinadas ao fundo $f$:

$$PU_f = \frac{\sum_{j=1}^{N} M_{j, f} \cdot P_j}{A_f}$$

onde $P_j$ é o preço unitário da execução $j$ e $A_f$ é a cota total do fundo $f$.

### 2.2. A Escolha do Erro Quadrático Médio (MSE)
Optou-se estritamente pela penalização baseada no **Erro Quadrático Médio (MSE)** em detrimento do Desvio Absoluto Médio (MAE):

$$\text{MSE} = \frac{1}{7} \sum_{f=1}^{7} (PU_f - PU_{\text{global}})^2$$

**Justificativa Teórica e Fiduciária:**
- O portfólio possui alta assimetria de tamanho: o Fundo 1 detém 47.000 ações (47%), enquanto o Fundo 7 detém apenas 200 ações (0,2%).
- Fundos pequenos são hiper-sensíveis: pequenas flutuações de alocação provocam variações bruscas no seu PU médio.
- Enquanto o MAE possui penalidade linear (tolerando que um fundo sofra grande desvio se a média geral for baixa), o **MSE eleva o desvio ao quadrado**, gerando uma punição desproporcional para outliers. Isso força o Algoritmo Genético a proteger ativamente os fundos menores (Fundos 6 e 7).

### 2.3. Formulação da Função Fitness
Para permitir a maximização evolutiva e eliminar singularidades:

$$\text{Fitness} = \frac{1}{1 + \text{MSE}}$$

- Para a solução ideal ($\text{MSE} = 0$), o $\text{Fitness} = 1{,}000000$ (100%).
- Quanto maior o erro, menor o fitness, mantendo o domínio estritamente em $(0, 1]$.

### 2.4. Validação Empírica: Estudo Comparativo MSE vs. MAE
Para comprovar cientificamente a superioridade do modelo baseado em Erro Quadrático Médio, implementamos o script `benchmark_mse_vs_mae.py`, comparando ambas as funções em múltiplos cenários e analisando a convergência e a dispersão dos preços médios por fundo.

![Estudo Comparativo MSE vs MAE](comparativo_mse_vs_mae.png)

**Principais Conclusões do Benchmark Empírico:**
1. **Proteção Rigorosa dos Fundos Menores (Painel A):**
   No modelo otimizado por MAE (laranja), o Fundo 7 (200 cotas) sofreu um desvio de PU superior a **R$ 0,0022**. Sob o modelo MSE (azul), a penalidade quadrática forçou o algoritmo a balancear os lotes, reduzindo o desvio do Fundo 7 pela metade (**< R$ 0,0010**).
2. **Velocidade de Convergência (Painel B):**
   A paisagem estritamente convexa do MSE elimina platôs e empates no torneio de seleção, permitindo que a população atinja erros mínimos em menos gerações do que a busca sob MAE.
3. **Controle de Variância e Eliminação de Outliers (Painel C):**
   Em uma análise estatística consolidada abrangendo 105 alocações de fundos em 15 cenários de teste, o modelo com MSE proporcionou uma **redução de 43,7% na variância dos desvios**. O boxplot evidencia que o modelo com MAE acumula uma cauda longa de *outliers* com desvios expressivos de até R$ 0,0040, enquanto o modelo com MSE compacta a distribuição e suprime distorções extremas.

---

## 3. Desafios Enfrentados e Decisões de Arquitetura

### Desafio 1: O Espaço de Busca Restrito e a Decomposição Pro-Rata

#### 1. A Inviabilidade de Abordagens Aleatórias Ingênuas
Em cenários com até 1.000 execuções e 7 fundos de investimento, a alocação requer preencher uma matriz inteira $M \in \mathbb{N}^{N \times 7}$ ($7.000$ variáveis de decisão) sujeita a um sistema rígido de restrições de igualdade em duas dimensões simultâneas:
1. **Conservação por Execução (Soma das Linhas):** $\sum_{f=1}^{7} M_{j, f} = Q_j, \quad \forall j \in \{1, \dots, N\}$ (nenhuma ação pode ser criada ou omitida de um lote).
2. **Conservação por Fundo (Soma das Colunas):** $\sum_{j=1}^{N} M_{j, f} = A_f, \quad \forall f \in \{1, \dots, 7\}$ (cada fundo deve receber exatamente sua cota contratada).

Se um Algoritmo Genético convencional gerasse ou mutasse indivíduos por valores inteiros pseudoaleatórios, a probabilidade de satisfazer simultaneamente todas as somas de linhas e colunas seria matematicamente nula ($P \approx 0$). Penalizar soluções inviáveis na função fitness levaria o algoritmo a ficar preso em regiões sem indivíduos viáveis (*death penalty* ineficaz).

---

#### 2. A Solução Arquitetural: Decomposição em Piso Inteiro Garantido e Ações Residuais

Para contornar a inviabilidade e acelerar drasticamente a convergência, a matriz de decisão é estruturada pela soma de duas parcelas desacopladas:

$$M_{j, f} = \text{Base}_{j, f} + X_{j, f}$$

##### A) O Piso Inteiro Garantido ($\text{Base}_{j, f}$)
Em teoria de finanças, a divisão perfeitamente equitativa (*pro-rata*) de um lote $j$ entregaria a cada carteira a sua proporção ideal:
$$\text{Cota Teórica}_{j, f} = Q_j \times \frac{A_f}{100.000}$$
Se a custódia permitisse frações contínuas de ações, todos os fundos receberiam exatamente essa fatia em cada lote, e o Preço Médio Ponderado ($PU_f$) de todas as carteiras seria **rigorosamente idêntico ao benchmark de R$ 10,48** (erro zero absoluto).

Contudo, como as regras de bolsa exigem **quantidades estritamente inteiras**, extraímos a parte inteira garantida via função piso (*floor*):
$$\text{Base}_{j, f} = \left\lfloor Q_j \times \frac{A_f}{100.000} \right\rfloor$$

* **O Comportamento nos Fundos Grandes:** Para o Fundo 1 (47.000 ações) e Fundo 2 (23.000 ações), o piso aloca diretamente mais de 98% a 99% das ações de forma proporcional, mantendo o PU médio dessas carteiras imediatamente colado em R$ 10,48.
* **O Fenômeno do Truncamento nos Fundos Pequenos:** O Fundo 7 detém apenas 200 cotas ($0{,}2\%$ da ordem). Em execuções com menos de 500 ações (a grande maioria dos lotes no pregão), a cota teórica resulta em frações menores que a unidade (ex: $26 \times 0{,}002 = 0{,}052$ ações). Ao aplicar o piso $\lfloor 0{,}052 \rfloor$, o resultado é **0 ações**.
  * *Impacto Empírico Real:* No Cenário 1 (554 execuções), o Fundo 7 recebe apenas 1 ação na base pura (uma execução grande negociada no pico de R$ 12,32). Seu PU provisório dispara para R$ 12,32 (+R$ 1,84 de distorção), ficando com **199 cotas pendentes**.

##### B) As Ações Residuais ($X_{j, f}$) e o Papel do Algoritmo Genético
O descarte das partes decimais no truncamento gera duas grandezas residuais complementares:
1. **Resíduo por Execução ($R_j$):** Quantidade de ações que sobraram no lote $j$ após a distribuição da base inteira:
   $$R_j = Q_j - \sum_{f=1}^{7} \text{Base}_{j, f} \quad \implies \quad R_j \in \{0, 1, 2, 3, 4, 5, 6\}$$
2. **Saldo Pendente por Fundo ($\Delta_f$):** Quantidade de ações que faltam para o fundo $f$ completar sua cota:
   $$\Delta_f = A_f - \sum_{j=1}^{N} \text{Base}_{j, f}$$

Pela propriedade de conservação da soma:
$$\sum_{j=1}^{N} R_j = \sum_{f=1}^{7} \Delta_f$$

A soma de todas as ações residuais dos lotes é **exatamente igual** à soma das ações que faltam para os fundos.

**Onde o Algoritmo Genético Atua:**
O AG é liberado do fardo de alocar 100.000 ações do zero e foca sua busca combinatória exclusivamente na **matriz residual $X_{j, f}$**:
- Ele decide estrategicamente quais lotes residuais com preços abaixo de R$ 10,48 devem ser entregues ao Fundo 7 para anular a execução cara de R$ 12,32 e puxar a média dele de volta para os R$ 10,48.
- Garante que todo indivíduo gerado nasça **100% factível**, com restrições de linhas e colunas satisfeitas por construção, permitindo que as gerações evolutivas foquem integralmente no refinamento cirúrgico do PU.

### Desafio 2: Operadores Genéticos que Preservam a Viabilidade
* **Crossover de Um Ponto com Reparo Conservativo:** 
  O corte transversal preserva a integridade de todas as linhas (execuções). As eventuais sobras nas colunas dos fundos são reparadas em tempo linear transferindo cotas dos fundos superavitários para os deficitários.
* **Mutação por Swaps Alternantes $2 \times 2$ (Ciclos Fechados):**
  A mutação escolhe dois lotes com preços diferentes ($j_1, j_2$) e dois fundos ($f_1, f_2$) e transfere cotas em circuito fechado:
  $$\Delta M_{j_1, f_1} = -\delta, \quad \Delta M_{j_1, f_2} = +\delta, \quad \Delta M_{j_2, f_2} = -\delta, \quad \Delta M_{j_2, f_1} = +\delta$$
  Essa operação altera o financeiro e o preço médio dos fundos **sem alterar nenhuma soma de linha e nenhuma soma de coluna**, preservando 100% da factibilidade sem custo de reparo.

---

## 4. Tecnologias Utilizadas

* **Linguagem:** Python 3.12 (robusta, tipada e com ampla adoção no mercado financeiro).
* **NumPy:** Vetorização de alta performance para cálculo matricial de fitness e operadores genéticos, permitindo tempos de convergência na ordem de milissegundos por geração.
* **Pandas:** Estruturação, manipulação e consolidação das massas de dados dos 500 cenários.
* **OpenPyXL:** Engine para geração de planilhas Excel estruturadas e auditáveis.

---

## 5. Resultados Obtidos e Conclusões

O algoritmo foi validado nos diferentes perfis de distribuição presentes na base de dados (concentrada, dispersa, aleatória, uniforme, exponencial e normal) e com números de execuções variando de 14 a 1.000 lotes:

* **SLA de Performance:** Processamento médio de **~0,20 segundos por cenário**, permitindo rodar os 500 cenários em menos de 2 minutos.
* **Aderência de Preço Médio:** 
  - **MSE Médio:** Na ordem de $10^{-7}$ a $10^{-6}$.
  - **Desvio Máximo Médio por Fundo:** Menor que **R$ 0,002** (menos de 2 décimos de centavo de real).
  - **Fitness:** $\approx 0{,}999999$ a $1{,}000000$.

### Sugestões de Melhoria para Produção:
1. **Minimização de Boletas (Taxas de Custódia):** Adicionar um termo de penalização secundário no fitness para preferir soluções com menor número de fragmentos, reduzindo custos de liquidação na B3.
2. **Paralelização Multiprocessada:** Utilizar `concurrent.futures.ProcessPoolExecutor` para distribuir os 500 cenários nos múltiplos núcleos da CPU da mesa, reduzindo o tempo de execução para menos de 30 segundos.
