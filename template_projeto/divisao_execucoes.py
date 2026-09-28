"""
Desafio Técnico: Divisão de Execuções entre Fundos via Algoritmo Genético
Empresa: Bahia Asset Management
Autor: Ryan Dias

Este script implementa um Algoritmo Genético para solucionar o problema de Fair Allocation
(rateio justo) de ordens institucionais fracionadas em N execuções entre 7 fundos de investimento.

Objetivo:
    Minimizar o Erro Quadrático Médio (MSE) entre o Preço Unitário Médio (PU) de cada fundo
    e o PU Médio Global da ordem (R$ 10,48), respeitando rigorosamente alocações inteiras.
"""

import os
import time
import random
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


# ==============================================================================
# 1. PARÂMETROS E HIPERPARÂMETROS DO ALGORITMO GENÉTICO
# ==============================================================================
POP_SIZE = 25            # Tamanho da população de soluções por cenário
GERACOES = 35            # Número de gerações evolutivas
TAXA_CROSSOVER = 0.70    # Probabilidade de recombinação entre dois pais
TAXA_MUTACAO = 0.50      # Probabilidade de mutação em um indivíduo
ELITISMO_COUNT = 2       # Quantidade dos melhores indivíduos mantidos intactos
TORNEIO_SIZE = 3         # Quantidade de competidores no torneio de seleção


# ==============================================================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================
def carregar_dados(caminho_alocacao: str, caminho_execucoes: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Carrega os arquivos CSV de alocação dos fundos e massa de execuções.
    
    Retorna:
        df_aloc: DataFrame com as cotas de cada fundo.
        df_exec: DataFrame com todas as execuções dos 500 cenários.
        nomes_fundos: Lista/Array com o nome dos 7 fundos.
        cotas_fundos: Array numpy com a quantidade de ações requerida por fundo.
    """
    df_aloc = pd.read_csv(caminho_alocacao)
    df_exec = pd.read_csv(caminho_execucoes)
    
    nomes_fundos = df_aloc['fundo'].to_numpy()
    cotas_fundos = df_aloc['quantidade'].to_numpy(dtype=np.int64)
    
    return df_aloc, df_exec, nomes_fundos, cotas_fundos


# ==============================================================================
# 3. FUNÇÃO FITNESS (AVALIAÇÃO BASEADA EM ERRO QUADRÁTICO MÉDIO)
# ==============================================================================
def calcular_fitness(
    matriz_alocacao: np.ndarray, 
    precos: np.ndarray, 
    cotas_fundos: np.ndarray, 
    pu_global: float
) -> Tuple[float, float, np.ndarray]:
    """
    Avalia a qualidade (fitness) de uma solução proposta.
    
    Etapas:
    1. Calcula o financeiro total de cada fundo: sum(qtd_alocada * preco).
    2. Calcula o Preço Unitário Médio (PU) de cada fundo: financeiro / cota_fundo.
    3. Calcula o Erro Quadrático Médio (MSE): média de (PU_fundo - PU_global)^2.
    4. Converte o MSE para a pontuação de aptidão: Fitness = 1.0 / (1.0 + MSE).
    
    Retorna:
        fitness: Escalar em (0, 1] onde 1.0 é a perfeição.
        mse: Erro quadrático médio.
        pu_fundos: Vetor com o PU médio alcançado por cada um dos 7 fundos.
    """
    # Financeiro alocado por fundo: produto escalar entre linhas e preços
    financeiro_fundos = (matriz_alocacao * precos[:, None]).sum(axis=0)
    
    # Preço Médio (PU) por fundo
    pu_fundos = financeiro_fundos / cotas_fundos
    
    # Erro Quadrático Médio (MSE) entre os fundos e o benchmark global
    mse = np.mean((pu_fundos - pu_global) ** 2)
    
    # Função Fitness inversamente proporcional ao erro
    fitness = 1.0 / (1.0 + mse)
    
    return fitness, mse, pu_fundos


# ==============================================================================
# 4. INICIALIZAÇÃO DA POPULAÇÃO (CRIAÇÃO DE INDIVÍDUOS VIÁVEIS)
# ==============================================================================
def criar_individuo(
    q_execs: np.ndarray, 
    cotas_fundos: np.ndarray, 
    props_fundos: np.ndarray
) -> np.ndarray:
    """
    Gera um indivíduo (matriz N x 7) que respeita 100% das restrições de integridade:
    - Linhas somam exatamente a quantidade da execução correspondente.
    - Colunas somam exatamente as cotas requeridas pelos 7 fundos.
    
    Estratégia:
    - Aloca a base inteira proporcional: floor(Q_j * proporcao_fundo).
    - Distribui os resíduos inteiros (de 0 a 5 ações por execução) de forma estocástica
      entre os fundos que ainda precisam de cotas.
    """
    n_execs = len(q_execs)
    n_fundos = len(cotas_fundos)
    
    # 1. Base inteira proporcional
    quotas = np.outer(q_execs, props_fundos)
    matriz = np.floor(quotas).astype(np.int64)
    
    # 2. Resíduos que sobraram em cada execução e quanto falta para cada fundo
    rem_linhas = (q_execs - matriz.sum(axis=1)).astype(np.int64)
    rem_colunas = (cotas_fundos - matriz.sum(axis=0)).astype(np.int64)
    
    # 3. Embaralha a ordem das execuções para gerar diversidade na população inicial
    ordem_execs = list(range(n_execs))
    random.shuffle(ordem_execs)
    
    for j in ordem_execs:
        r = rem_linhas[j]
        while r > 0:
            candidatos = [f for f in range(n_fundos) if rem_colunas[f] > 0]
            if not candidatos:
                break
            
            # Sorteia um dos fundos com saldo pendente
            f = random.choice(candidatos)
            delta = min(r, rem_colunas[f])
            
            matriz[j, f] += delta
            r -= delta
            rem_colunas[f] -= delta
            rem_linhas[j] -= delta
            
    return matriz


# ==============================================================================
# 5. OPERADORES GENÉTICOS: CROSSOVER, MUTAÇÃO E REPARO
# ==============================================================================
def crossover(pai1: np.ndarray, pai2: np.ndarray, cotas_fundos: np.ndarray) -> np.ndarray:
    """
    Realiza o Crossover de Um Ponto entre duas matrizes viáveis.
    
    Como as primeiras 'ponto' linhas vêm do Pai 1 e as demais do Pai 2:
    - As somas de cada linha permanecem 100% exatas (respeitam a quantidade da execução).
    - As somas das colunas podem sofrer pequenos desvios, que são reparados
      imediatamente transferindo cotas dos fundos excedentes para os deficitários.
    """
    n_execs = pai1.shape[0]
    ponto = random.randint(1, n_execs - 1)
    
    filho = np.vstack([pai1[:ponto], pai2[ponto:]]).copy()
    
    # Reparo de colunas (mantendo integridade das cotas de cada fundo)
    col_diffs = (cotas_fundos - filho.sum(axis=0)).astype(np.int64)
    
    max_iter = 60
    while max_iter > 0:
        max_iter -= 1
        pos = [f for f in range(7) if col_diffs[f] > 0] # precisa receber
        neg = [f for f in range(7) if col_diffs[f] < 0] # precisa doar
        
        if not pos or not neg:
            break
            
        f_rec = pos[0]
        f_doa = neg[0]
        needed = min(col_diffs[f_rec], -col_diffs[f_doa])
        
        # Encontra execuções onde o fundo doador possui ações alocadas
        linhas_candidatas = np.where(filho[:, f_doa] > 0)[0]
        if len(linhas_candidatas) == 0:
            break
            
        j = random.choice(linhas_candidatas)
        delta = min(needed, filho[j, f_doa])
        
        filho[j, f_doa] -= delta
        filho[j, f_rec] += delta
        col_diffs[f_doa] += delta
        col_diffs[f_rec] -= delta
        
    return filho


def mutacao(matriz: np.ndarray, num_trocas: int = 5) -> np.ndarray:
    """
    Aplica Mutação por Swaps Alternantes 2x2:
    - Escolhe duas execuções j1 e j2 com preços diferentes.
    - Escolhe dois fundos f1 e f2.
    - Realiza uma transferência em circuito fechado:
        M[j1, f1] -= delta;  M[j1, f2] += delta
        M[j2, f2] -= delta;  M[j2, f1] += delta
        
    Vantagem Matemática:
    - Preserva estritamente a soma de todas as linhas e colunas.
    - Modifica a distribuição dos preços médios entre os fundos.
    """
    filho = matriz.copy()
    n_execs = matriz.shape[0]
    
    if n_execs < 2:
        return filho
        
    for _ in range(num_trocas):
        j1, j2 = random.sample(range(n_execs), 2)
        
        f1_cands = [f for f in range(7) if filho[j1, f] > 0]
        f2_cands = [f for f in range(7) if filho[j2, f] > 0]
        
        if not f1_cands or not f2_cands:
            continue
            
        f1 = random.choice(f1_cands)
        f2 = random.choice(f2_cands)
        
        if f1 == f2:
            continue
            
        delta_max = min(filho[j1, f1], filho[j2, f2])
        if delta_max <= 0:
            continue
            
        delta = random.randint(1, min(delta_max, 5))
        
        filho[j1, f1] -= delta
        filho[j1, f2] += delta
        filho[j2, f2] -= delta
        filho[j2, f1] += delta
        
    return filho


# ==============================================================================
# 6. MOTOR EVOLUTIVO PRINCIPAL (POR CENÁRIO)
# ==============================================================================
def otimizar_cenario(
    cenario_id: int,
    df_cenario: pd.DataFrame, 
    cotas_fundos: np.ndarray, 
    props_fundos: np.ndarray
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Executa o Algoritmo Genético para um cenário específico de execuções.
    
    Retorna:
        melhor_matriz: Matriz de alocação com o menor erro encontrado.
        melhor_fit: Fitness máximo atingido.
        melhor_mse: Erro quadrático médio mínimo.
        melhor_pu: Vetor com o PU médio final dos 7 fundos.
    """
    q_execs = df_cenario['quantidade'].to_numpy(dtype=np.int64)
    precos = df_cenario['preco'].to_numpy(dtype=np.float64)
    pu_global = (q_execs * precos).sum() / cotas_fundos.sum()
    
    # 1. População Inicial
    populacao = [criar_individuo(q_execs, cotas_fundos, props_fundos) for _ in range(POP_SIZE)]
    avaliacoes = [calcular_fitness(ind, precos, cotas_fundos, pu_global) for ind in populacao]
    
    # 2. Laço de Gerações
    for g in range(GERACOES):
        # Ordena indivíduos por fitness decrescente (Elitismo)
        ordenados = sorted(zip(avaliacoes, populacao), key=lambda x: x[0][0], reverse=True)
        
        nova_pop = []
        # Preserva os melhores indivíduos intactos
        for i in range(ELITISMO_COUNT):
            nova_pop.append(ordenados[i][1].copy())
            
        # Gera o restante da nova geração
        while len(nova_pop) < POP_SIZE:
            # Seleção por Torneio
            candidatos_torneio1 = random.sample(ordenados, TORNEIO_SIZE)
            pai1 = max(candidatos_torneio1, key=lambda x: x[0][0])[1]
            
            candidatos_torneio2 = random.sample(ordenados, TORNEIO_SIZE)
            pai2 = max(candidatos_torneio2, key=lambda x: x[0][0])[1]
            
            # Crossover
            if random.random() < TAXA_CROSSOVER:
                filho = crossover(pai1, pai2, cotas_fundos)
            else:
                filho = pai1.copy()
                
            # Mutação
            if random.random() < TAXA_MUTACAO:
                filho = mutacao(filho, num_trocas=random.randint(1, 8))
                
            nova_pop.append(filho)
            
        populacao = nova_pop
        avaliacoes = [calcular_fitness(ind, precos, cotas_fundos, pu_global) for ind in populacao]
        
    # Extrai a melhor solução final encontrada
    melhor_aval, melhor_matriz = max(zip(avaliacoes, populacao), key=lambda x: x[0][0])
    melhor_fit, melhor_mse, melhor_pu = melhor_aval
    
    return melhor_matriz, melhor_fit, melhor_mse, melhor_pu


# ==============================================================================
# 7. EXECUÇÃO EM MASSA E GERAÇÃO DOS ARQUIVOS DE SAÍDA (.XLSX)
# ==============================================================================
def processar_todos_os_cenarios(
    caminho_alocacao: str, 
    caminho_execucoes: str, 
    caminho_saida_excel: str,
    max_cenarios: int = None
):
    """
    Processa todos os cenários de execuções com o Algoritmo Genético e exporta
    o resultado para o arquivo Excel esperado pelo desafio.
    """
    print("=" * 80)
    print("INICIANDO RESOLUÇÃO DOS CENÁRIOS COM ALGORITMO GENÉTICO")
    print("=" * 80)
    
    df_aloc, df_exec, nomes_fundos, cotas_fundos = carregar_dados(caminho_alocacao, caminho_execucoes)
    props_fundos = cotas_fundos / cotas_fundos.sum()
    
    cenarios_disponiveis = sorted(df_exec['cenario_id'].unique())
    if max_cenarios is not None:
        cenarios_disponiveis = cenarios_disponiveis[:max_cenarios]
        
    total_cenarios = len(cenarios_disponiveis)
    print(f"Total de cenários a processar: {total_cenarios}")
    print(f"Fundos: {list(nomes_fundos)}")
    print(f"Cotas: {list(cotas_fundos)} (Total: {cotas_fundos.sum()})\n")
    
    lista_dfs_saida = []
    lista_metricas_cenarios = []
    
    tempo_inicio_global = time.time()
    
    for idx, c_id in enumerate(cenarios_disponiveis, start=1):
        t0 = time.time()
        df_cenario = df_exec[df_exec['cenario_id'] == c_id].copy().reset_index(drop=True)
        
        melhor_matriz, melhor_fit, melhor_mse, melhor_pu = otimizar_cenario(
            c_id, df_cenario, cotas_fundos, props_fundos
        )
        t_cenario = time.time() - t0
        
        # Constrói o DataFrame do rateio deste cenário
        df_res = df_cenario[['cenario_id', 'execucao_id', 'quantidade', 'preco', 'tipo_distribuicao']].copy()
        for i, nome_fundo in enumerate(nomes_fundos):
            df_res[nome_fundo] = melhor_matriz[:, i]
            
        lista_dfs_saida.append(df_res)
        
        # Registra métricas de qualidade
        pu_global = (df_cenario['quantidade'] * df_cenario['preco']).sum() / cotas_fundos.sum()
        lista_metricas_cenarios.append({
            'cenario_id': c_id,
            'mse': melhor_mse,
            'fitness': melhor_fit,
            'pu_global': pu_global,
            'max_desvio_fundo': np.max(np.abs(melhor_pu - pu_global)),
            'tempo_segundos': t_cenario
        })
        
        if idx % 10 == 0 or idx == total_cenarios or idx <= 5:
            desvio_max = np.max(np.abs(melhor_pu - pu_global))
            print(f"[{idx:03d}/{total_cenarios:03d}] Cenário {c_id:03d} | MSE: {melhor_mse:.8f} | Fit: {melhor_fit:.6f} | Max Desvio: R$ {desvio_max:.4f} | Tempo: {t_cenario:.2f}s")
            
    tempo_total = time.time() - tempo_inicio_global
    print("\n" + "=" * 80)
    print(f"PROCESSAMENTO CONCLUÍDO EM {tempo_total:.2f} SEGUNDOS ({tempo_total/60:.2f} MINUTOS)!")
    print("=" * 80)
    
    # Consolida os dados e salva em Excel
    print(f"Consolidando planilha Excel: {caminho_saida_excel} ...")
    df_consolidado = pd.concat(lista_dfs_saida, ignore_index=True)
    df_metricas = pd.DataFrame(lista_metricas_cenarios)
    
    with pd.ExcelWriter(caminho_saida_excel, engine='openpyxl') as writer:
        df_consolidado.to_excel(writer, sheet_name='Divisao_Execucoes', index=False)
        df_metricas.to_excel(writer, sheet_name='Metricas_Cenarios', index=False)
        
    print("Arquivo Excel gerado com sucesso!")
    print(f"MSE Médio Geral: {df_metricas['mse'].mean():.8f}")
    print(f"Desvio Máximo Médio por Fundo: R$ {df_metricas['max_desvio_fundo'].mean():.4f}")


# ==============================================================================
# PONTO DE ENTRADA
# ==============================================================================
if __name__ == '__main__':
    # Define caminhos absolutos baseados no diretório do projeto
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    CAMINHO_ALOC = os.path.join(BASE_DIR, 'alocacao_ordem.csv')
    CAMINHO_EXEC = os.path.join(BASE_DIR, 'massa_execucoes_500_cenarios.csv')
    CAMINHO_OUT_EXCEL = os.path.join(BASE_DIR, 'resultado_divisao.xlsx')
    processar_todos_os_cenarios(
        caminho_alocacao=CAMINHO_ALOC,
        caminho_execucoes=CAMINHO_EXEC,
        caminho_saida_excel=CAMINHO_OUT_EXCEL,
        max_cenarios=None  # Processa todos os 500 cenários
    )
    
    # Salva também uma cópia no template_projeto
    caminho_template_excel = os.path.join(BASE_DIR, 'template_projeto', 'resultado_divisao.xlsx')
    if os.path.exists(CAMINHO_OUT_EXCEL):
        import shutil
        shutil.copyfile(CAMINHO_OUT_EXCEL, caminho_template_excel)
        print(f"Cópia sincronizada em: {caminho_template_excel}")

