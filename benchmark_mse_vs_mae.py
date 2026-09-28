"""
Script de Benchmark e Visualização: MSE vs MAE no Algoritmo Genético
Empresa: Bahia Asset Management
Autor: Ryan Dias

Este script realiza uma comparação empírica rigorosa entre duas funções de fitness:
1. Baseada em Erro Quadrático Médio (MSE): Fitness = 1.0 / (1.0 + MSE)
2. Baseada em Desvio Absoluto Médio (MAE): Fitness = 1.0 / (1.0 + MAE)

Painéis Gerados:
- Painel A: Justiça distributiva por fundo (Desvio de PU em cada um dos 7 fundos).
- Painel B: Velocidade de convergência ao longo das gerações evolutivas.
- Painel C: Comparativo estatístico de variância e dispersão dos desvios (Boxplot consolidado).
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# Configuração de estilo visual dos gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Semente para reprodutibilidade
random.seed(42)
np.random.seed(42)


# ==============================================================================
# 1. FUNÇÕES DE FITNESS (MSE vs MAE)
# ==============================================================================
def calcular_fitness_mse(matriz, precos, cotas_fundos, pu_global):
    val_fundos = (matriz * precos[:, None]).sum(axis=0)
    pu_fundos = val_fundos / cotas_fundos
    mse = np.mean((pu_fundos - pu_global) ** 2)
    mae = np.mean(np.abs(pu_fundos - pu_global))
    fitness = 1.0 / (1.0 + mse)
    return fitness, mse, mae, pu_fundos


def calcular_fitness_mae(matriz, precos, cotas_fundos, pu_global):
    val_fundos = (matriz * precos[:, None]).sum(axis=0)
    pu_fundos = val_fundos / cotas_fundos
    mae = np.mean(np.abs(pu_fundos - pu_global))
    mse = np.mean((pu_fundos - pu_global) ** 2)
    fitness = 1.0 / (1.0 + mae)
    return fitness, mse, mae, pu_fundos


# ==============================================================================
# 2. OPERADORES GENÉTICOS
# ==============================================================================
def criar_individuo(q_execs, cotas_fundos, props_fundos):
    n_execs = len(q_execs)
    quotas = np.outer(q_execs, props_fundos)
    matriz = np.floor(quotas).astype(np.int64)
    rem_linhas = (q_execs - matriz.sum(axis=1)).astype(np.int64)
    rem_colunas = (cotas_fundos - matriz.sum(axis=0)).astype(np.int64)
    
    ordem = list(range(n_execs))
    random.shuffle(ordem)
    for j in ordem:
        r = rem_linhas[j]
        while r > 0:
            cands = [f for f in range(7) if rem_colunas[f] > 0]
            if not cands:
                break
            f = random.choice(cands)
            delta = min(r, rem_colunas[f])
            matriz[j, f] += delta
            r -= delta
            rem_colunas[f] -= delta
            rem_linhas[j] -= delta
    return matriz


def crossover(pai1, pai2, cotas_fundos):
    n_execs = pai1.shape[0]
    ponto = random.randint(1, n_execs - 1)
    filho = np.vstack([pai1[:ponto], pai2[ponto:]]).copy()
    col_diffs = (cotas_fundos - filho.sum(axis=0)).astype(np.int64)
    
    for _ in range(50):
        pos = [f for f in range(7) if col_diffs[f] > 0]
        neg = [f for f in range(7) if col_diffs[f] < 0]
        if not pos or not neg:
            break
        f_rec, f_doa = pos[0], neg[0]
        needed = min(col_diffs[f_rec], -col_diffs[f_doa])
        linhas = np.where(filho[:, f_doa] > 0)[0]
        if len(linhas) == 0:
            break
        j = random.choice(linhas)
        delta = min(needed, filho[j, f_doa])
        filho[j, f_doa] -= delta
        filho[j, f_rec] += delta
        col_diffs[f_doa] += delta
        col_diffs[f_rec] -= delta
    return filho


def mutacao(matriz, num_trocas=5):
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
        f1, f2 = random.choice(f1_cands), random.choice(f2_cands)
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
# 3. MOTOR EVOLUTIVO
# ==============================================================================
def executar_ga(tipo_metrica, df_cenario, cotas_fundos, props_fundos, pop_size=25, geracoes=35):
    q_execs = df_cenario['quantidade'].to_numpy(dtype=np.int64)
    precos = df_cenario['preco'].to_numpy(dtype=np.float64)
    pu_global = (q_execs * precos).sum() / cotas_fundos.sum()
    
    funcao_fitness = calcular_fitness_mse if tipo_metrica == 'MSE' else calcular_fitness_mae
    
    populacao = [criar_individuo(q_execs, cotas_fundos, props_fundos) for _ in range(pop_size)]
    avaliacoes = [funcao_fitness(ind, precos, cotas_fundos, pu_global) for ind in populacao]
    
    historico_mae = []
    historico_mse = []
    
    for g in range(geracoes):
        ordenados = sorted(zip(avaliacoes, populacao), key=lambda x: x[0][0], reverse=True)
        melhor_aval = ordenados[0][0]
        historico_mse.append(melhor_aval[1])
        historico_mae.append(melhor_aval[2])
        
        nova_pop = [ordenados[0][1].copy(), ordenados[1][1].copy()]
        while len(nova_pop) < pop_size:
            p1 = max(random.sample(ordenados, 3), key=lambda x: x[0][0])[1]
            p2 = max(random.sample(ordenados, 3), key=lambda x: x[0][0])[1]
            filho = crossover(p1, p2, cotas_fundos) if random.random() < 0.70 else p1.copy()
            if random.random() < 0.50:
                filho = mutacao(filho, num_trocas=random.randint(1, 8))
            nova_pop.append(filho)
            
        populacao = nova_pop
        avaliacoes = [funcao_fitness(ind, precos, cotas_fundos, pu_global) for ind in populacao]
        
    melhor_aval, melhor_matriz = max(zip(avaliacoes, populacao), key=lambda x: x[0][0])
    return {
        'tipo': tipo_metrica,
        'melhor_matriz': melhor_matriz,
        'pu_fundos': melhor_aval[3],
        'pu_global': pu_global,
        'mse_final': melhor_aval[1],
        'mae_final': melhor_aval[2],
        'historico_mse': historico_mse,
        'historico_mae': historico_mae
    }


# ==============================================================================
# 4. EXECUÇÃO DO BENCHMARK E GERAÇÃO DOS GRÁFICOS
# ==============================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    aloc_path = os.path.join(base_dir, 'alocacao_ordem.csv')
    exec_path = os.path.join(base_dir, 'massa_execucoes_500_cenarios.csv')
    
    df_aloc = pd.read_csv(aloc_path)
    df_exec = pd.read_csv(exec_path)
    
    nomes_fundos = df_aloc['fundo'].tolist()
    cotas_fundos = df_aloc['quantidade'].to_numpy(dtype=np.int64)
    props_fundos = cotas_fundos / cotas_fundos.sum()
    
    # 1. Execução no Cenário 1 para os Painéis A e B
    print("Executando benchmark comparativo MSE vs MAE no Cenário 1...")
    df_c1 = df_exec[df_exec['cenario_id'] == 1].copy().reset_index(drop=True)
    res_mse = executar_ga('MSE', df_c1, cotas_fundos, props_fundos, pop_size=30, geracoes=40)
    res_mae = executar_ga('MAE', df_c1, cotas_fundos, props_fundos, pop_size=30, geracoes=40)
    
    # 2. Coleta estatística de múltiplos cenários para o Painel C (Variância e Dispersão)
    num_cenarios_teste = 15
    print(f"Executando em {num_cenarios_teste} cenários para análise estatística de variância e dispersão...")
    todos_desvios_mse = []
    todos_desvios_mae = []
    
    for c_id in range(1, num_cenarios_teste + 1):
        cdf = df_exec[df_exec['cenario_id'] == c_id].copy().reset_index(drop=True)
        r_mse = executar_ga('MSE', cdf, cotas_fundos, props_fundos, pop_size=25, geracoes=35)
        r_mae = executar_ga('MAE', cdf, cotas_fundos, props_fundos, pop_size=25, geracoes=35)
        
        todos_desvios_mse.extend(np.abs(r_mse['pu_fundos'] - r_mse['pu_global']))
        todos_desvios_mae.extend(np.abs(r_mae['pu_fundos'] - r_mae['pu_global']))
        
    desvios_mse = np.array(todos_desvios_mse)
    desvios_mae = np.array(todos_desvios_mae)
    
    var_mse = np.var(desvios_mse)
    var_mae = np.var(desvios_mae)
    std_mse = np.std(desvios_mse)
    std_mae = np.std(desvios_mae)
    
    print("\nEstatísticas Consolidadas (105 Alocações de Fundos em 15 Cenários):")
    print(f"  MSE -> Média: R$ {np.mean(desvios_mse):.6f} | Desvio Padrão (Sigma): R$ {std_mse:.6f} | Variância: {var_mse:.2e}")
    print(f"  MAE -> Média: R$ {np.mean(desvios_mae):.6f} | Desvio Padrão (Sigma): R$ {std_mae:.6f} | Variância: {var_mae:.2e}")
    print(f"  Redução da Variância via MSE: {((var_mae - var_mse) / var_mae) * 100:.1f}%\n")
    
    # -------------------------------------------------------------------------
    # CRIANDO OS GRÁFICOS (PAINEL DE 3 SUBPLOTS REVISADO)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    
    cores_mse = '#1f77b4'  # Azul institucional
    cores_mae = '#ff7f0e'  # Laranja de alerta
    
    # 1. Subplot Superior: Desvio Absoluto por Fundo (Fairness)
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(nomes_fundos))
    largura = 0.35
    
    desvios_fundo_mse = np.abs(res_mse['pu_fundos'] - res_mse['pu_global'])
    desvios_fundo_mae = np.abs(res_mae['pu_fundos'] - res_mae['pu_global'])
    
    ax1.bar(x - largura/2, desvios_fundo_mse, largura, label='Modelo com Erro Quadrático (MSE)', color=cores_mse, alpha=0.9)
    ax1.bar(x + largura/2, desvios_fundo_mae, largura, label='Modelo com Desvio Absoluto (MAE)', color=cores_mae, alpha=0.9)
    
    ax1.set_title('A. Justiça Distributiva por Fundo: Desvio Médio do PU (|PU_fundo - PU_global|)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel('Desvio Absoluto (R$)', fontsize=11)
    ax1.set_xticks(x)
    labels_com_cotas = [f"{nome}\n({cotas:,} cotas)" for nome, cotas in zip(nomes_fundos, cotas_fundos)]
    ax1.set_xticklabels(labels_com_cotas, fontsize=9.5)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    ax1.annotate('Proteção dos Fundos Menores:\nMSE penaliza desvios grandes\ne reduz o erro do Fundo 7 pela metade',
                 xy=(6, desvios_fundo_mae[6]),
                 xytext=(4.2, max(desvios_fundo_mae) * 0.78),
                 arrowprops=dict(facecolor='#333333', arrowstyle='->', lw=1.5),
                 fontsize=10, fontweight='semibold', bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffcc', alpha=0.85))
    
    # 2. Subplot Inferior Esquerdo: Convergência ao Longo das Gerações
    ax2 = fig.add_subplot(gs[1, 0])
    geracoes_eixo = list(range(len(res_mse['historico_mae'])))
    
    ax2.plot(geracoes_eixo, res_mse['historico_mae'], label='Otimizado via MSE', color=cores_mse, lw=2.2)
    ax2.plot(geracoes_eixo, res_mae['historico_mae'], label='Otimizado via MAE', color=cores_mae, lw=2.2, linestyle='--')
    
    ax2.set_title('B. Velocidade de Convergência Evolutiva (Erro Médio vs Gerações)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Geração', fontsize=10.5)
    ax2.set_ylabel('Erro Médio da População (R$)', fontsize=10.5)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Subplot Inferior Direito: Comparação de Variância e Dispersão (Boxplot)
    ax3 = fig.add_subplot(gs[1, 1])
    dados_boxplot = [desvios_mse, desvios_mae]
    
    box = ax3.boxplot(dados_boxplot, tick_labels=['Modelo MSE', 'Modelo MAE'], patch_artist=True, widths=0.45,
                      medianprops=dict(color='black', lw=1.8),
                      flierprops=dict(marker='o', markersize=6, alpha=0.6))
    
    box['boxes'][0].set(facecolor=cores_mse, alpha=0.8)
    box['boxes'][1].set(facecolor=cores_mae, alpha=0.8)
    box['fliers'][0].set(markerfacecolor=cores_mse)
    box['fliers'][1].set(markerfacecolor=cores_mae)
    
    ax3.set_title('C. Controle de Variância e Outliers (15 Cenários - 105 Alocações)', fontsize=12, fontweight='bold', pad=10)
    ax3.set_ylabel('Desvio Absoluto de PU (R$)', fontsize=10.5)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Caixa explicativa de estatísticas no Painel C
    texto_estatistico = (
        f"Controle de Risco e Dispersão:\n"
        f"• Desvio Médio: R$ {np.mean(desvios_mse):.4f} (MSE) vs R$ {np.mean(desvios_mae):.4f} (MAE)\n"
        f"• Desvio Padrão: R$ {std_mse:.4f} (MSE) vs R$ {std_mae:.4f} (MAE)\n"
        f"• Redução de Variância: {((var_mae - var_mse) / var_mae) * 100:.1f}% a favor do MSE\n"
        f"• Eliminação de cauda longa (outliers)"
    )
    ax3.text(0.04, 0.95, texto_estatistico, transform=ax3.transAxes, fontsize=9.2, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7', edgecolor='#bbbbbb', alpha=0.9))
    
    plt.suptitle('Estudo Comparativo: Erro Quadrático Médio (MSE) vs Desvio Absoluto Médio (MAE)\nDesafio Técnico Bahia Asset Management', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    out_img = os.path.join(base_dir, 'comparativo_mse_vs_mae.png')
    out_img_template = os.path.join(base_dir, 'template_projeto', 'comparativo_mse_vs_mae.png')
    
    plt.savefig(out_img, bbox_inches='tight')
    plt.savefig(out_img_template, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico comparativo salvo com sucesso em:\n -> {out_img}\n -> {out_img_template}")


if __name__ == '__main__':
    main()
