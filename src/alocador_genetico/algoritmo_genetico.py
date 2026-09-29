import random
from typing import Optional, Tuple, List
import numpy as np
from .config import ConfiguracaoAG


class AlgoritmoGenetico:
    """
    Otimizador evolutivo para alocação justa de execuções entre múltiplos fundos.
    O construtor recebe as definições do cenário uma única vez, simplificando os métodos internos.
    """

    def __init__(
        self,
        cotas_fundos: np.ndarray,
        qtd_execucoes: np.ndarray,
        precos: np.ndarray,
        configuracao: Optional[ConfiguracaoAG] = None
    ):
        self.cotas = cotas_fundos
        self.qtd_execucoes = qtd_execucoes
        self.precos = precos
        self.config = configuracao or ConfiguracaoAG()

        # Métricas globais do cenário
        self.total_acoes = self.cotas.sum()
        # Proporção ideal de cada carteira na ordem consolidada (ex: 47% para Fundo 1, 0,2% para Fundo 7)
        self.proporcoes = self.cotas / self.total_acoes

        # Preço Médio Ponderado Global de referência da ordem fiduciária (poderia preencher com R$10.48, mas é mais usual usar um cálculo dinâmico)
        self.pu_global = (self.qtd_execucoes * self.precos).sum() / self.total_acoes
        self.n_execucoes = len(self.qtd_execucoes)
        self.n_fundos = len(self.cotas)

        # gera a matriz teórica contínua (N x 7) via produto externo das quantidades pelos percentuais
        cotas_teoricas = np.outer(self.qtd_execucoes, self.proporcoes)
        
        # Piso inteiro: garante a maior fatia proporcional em números inteiros sem violar custódia
        self.alocacao_base = np.floor(cotas_teoricas).astype(np.int64)
        
        # Ações residuais em cada execução que sobraram após a distribuição da base inteira
        self.saldo_linhas_base = (self.qtd_execucoes - self.alocacao_base.sum(axis=1)).astype(np.int64)
        
        # Cotas pendentes que cada fundo ainda precisa receber para atingir sua meta contratada
        self.saldo_colunas_base = (self.cotas - self.alocacao_base.sum(axis=0)).astype(np.int64)

    def avaliar(self, individuo: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Calcula o Preço Unitário Médio por fundo e a nota de fitness via Erro Quadrático Médio."""
        # Multiplicação matricial com broadcast: pondera as ações alocadas pelo preço de cada execução
        financeiro_fundos = (individuo * self.precos[:, None]).sum(axis=0)

        pu_fundos = financeiro_fundos / self.cotas
        
        mse = float(np.mean((pu_fundos - self.pu_global) ** 2))

        aptidao = 1.0 / (1.0 + mse)
        return aptidao, mse, pu_fundos

    def criar_individuo(self) -> np.ndarray:
        """Gera um indivíduo 100% viável preenchendo a base inteira e sorteando os resíduos."""
        individuo = self.alocacao_base.copy()
        saldo_linhas = self.saldo_linhas_base.copy()
        saldo_colunas = self.saldo_colunas_base.copy()

        # Embaralha a ordem de visitação dos lotes para garantir diversidade inicial na população
        ordem_lotes = list(range(self.n_execucoes))
        random.shuffle(ordem_lotes)

        for j in ordem_lotes:
            sobra_lote = saldo_linhas[j]
            while sobra_lote > 0:
                # Filtra apenas carteiras que ainda possuem cotas pendentes para receber
                fundos_candidatos = [f for f in range(self.n_fundos) if saldo_colunas[f] > 0]
                if not fundos_candidatos:
                    break
                f = random.choice(fundos_candidatos)
                
                # Transfere o máximo possível de cotas residuais sem estourar o lote nem o fundo
                delta = min(sobra_lote, saldo_colunas[f])
                individuo[j, f] += delta
                sobra_lote -= delta
                saldo_colunas[f] -= delta
                saldo_linhas[j] -= delta

        return individuo

    def criar_populacao(self) -> List[np.ndarray]:
        """Inicializa a população com a quantidade configurada de indivíduos válidos."""
        return [self.criar_individuo() for _ in range(self.config.tamanho_populacao)]

    def crossover(self, pai1: np.ndarray, pai2: np.ndarray) -> np.ndarray:
        """Recombina duas soluções candidatas com corte transversal e reparo conservativo de cotas."""
        ponto_corte = random.randint(1, self.n_execucoes - 1)
        # Corte horizontal nas linhas da planilha: preserva a integridade de conservação de cada lote
        filho = np.vstack([pai1[:ponto_corte], pai2[ponto_corte:]]).copy()

        # Saldo de cotas por fundo na solução filha 
        diferenca_fundos = (self.cotas - filho.sum(axis=0)).astype(np.int64)
        limite_reparos = 60

        # Algoritmo de reparo linear: reequilibra as cotas dos fundos mantendo a soma das linhas intacta
        while limite_reparos > 0:
            limite_reparos -= 1
            deficitarios = [f for f in range(self.n_fundos) if diferenca_fundos[f] > 0]
            excedentes = [f for f in range(self.n_fundos) if diferenca_fundos[f] < 0]
            if not deficitarios or not excedentes:
                break

            f_rec = deficitarios[0]
            f_doa = excedentes[0]
            qtd_transferir = min(diferenca_fundos[f_rec], -diferenca_fundos[f_doa])

            # Localiza execuções onde o fundo doador possui ações alocadas disponíveis para repassar
            lotes_doador = np.where(filho[:, f_doa] > 0)[0]
            if len(lotes_doador) == 0:
                break

            j = random.choice(lotes_doador)
            delta = min(qtd_transferir, filho[j, f_doa])

            # Transfere as ações dentro da mesma execução j: a soma da linha permanece inalterada
            filho[j, f_doa] -= delta
            filho[j, f_rec] += delta
            diferenca_fundos[f_doa] += delta
            diferenca_fundos[f_rec] -= delta

        return filho

    def mutar(self, individuo: np.ndarray, num_trocas: int = 5) -> np.ndarray:
        """Aplica mutação por swaps 2x2 em circuito fechado, mantendo a integridade das somas."""
        mutante = individuo.copy()
        if self.n_execucoes < 2:
            return mutante

        for _ in range(num_trocas):
            # Sorteia dois lotes distintos de execução para explorar a diferença de preços
            j1, j2 = random.sample(range(self.n_execucoes), 2)
            cand_f1 = [f for f in range(self.n_fundos) if mutante[j1, f] > 0]
            cand_f2 = [f for f in range(self.n_fundos) if mutante[j2, f] > 0]

            if not cand_f1 or not cand_f2:
                continue

            f1 = random.choice(cand_f1)
            f2 = random.choice(cand_f2)
            if f1 == f2:
                continue

            # Garante que a quantidade transferida não resulte em valores negativos de custódia
            max_delta = min(mutante[j1, f1], mutante[j2, f2])
            if max_delta <= 0:
                continue

            delta = random.randint(1, min(max_delta, 5))
            
            # Swaps 2x2 em circuito fechado
            mutante[j1, f1] -= delta
            mutante[j1, f2] += delta
            mutante[j2, f2] -= delta
            mutante[j2, f1] += delta

        return mutante

    def executar(self) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """Executa as gerações do Algoritmo Genético e retorna a melhor alocação encontrada."""
        populacao = self.criar_populacao()
        avaliacoes = [self.avaliar(ind) for ind in populacao]

        for _ in range(self.config.geracoes):
            ordenados = sorted(zip(avaliacoes, populacao), key=lambda item: item[0][0], reverse=True)
            
            # Elitismo: preserva as melhores soluções diretamente para a geração seguinte
            nova_populacao = [ordenados[i][1].copy() for i in range(self.config.tamanho_elite)]

            while len(nova_populacao) < self.config.tamanho_populacao:
                # Seleção por Torneio: seleciona os pais mais aptos entre concorrentes sorteados
                p1 = max(random.sample(ordenados, self.config.tamanho_torneio), key=lambda x: x[0][0])[1]
                p2 = max(random.sample(ordenados, self.config.tamanho_torneio), key=lambda x: x[0][0])[1]

                filho = (
                    self.crossover(p1, p2)
                    if random.random() < self.config.taxa_crossover
                    else p1.copy()
                )

                if random.random() < self.config.taxa_mutacao:
                    filho = self.mutar(filho, num_trocas=random.randint(1, 8))

                nova_populacao.append(filho)

            populacao = nova_populacao
            avaliacoes = [self.avaliar(ind) for ind in populacao]

        # Extrai a solução com a melhor nota de aptidão (menor MSE) ao término das gerações
        melhor_avaliacao, melhor_solucao = max(zip(avaliacoes, populacao), key=lambda x: x[0][0])
        aptidao, mse, pu_fundos = melhor_avaliacao
        return melhor_solucao, mse, aptidao, pu_fundos
