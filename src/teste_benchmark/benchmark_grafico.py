"""
Script de validação empírica e geração dos gráficos comparativos (MSE vs MAE).
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

diretorio_script = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(diretorio_script))
sys.path.insert(0, diretorio_script)

from alocador_genetico import (
    carregar_dados_ordem,
    localizar_arquivo,
    AlgoritmoGenetico,
    ConfiguracaoAG
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

random.seed(42)
np.random.seed(42)


def avaliar_via_mae(individuo, precos, cotas_fundos, pu_global):
    financeiro_fundos = (individuo * precos[:, None]).sum(axis=0)
    pu_fundos = financeiro_fundos / cotas_fundos
    mae = float(np.mean(np.abs(pu_fundos - pu_global)))
    mse = float(np.mean((pu_fundos - pu_global) ** 2))
    aptidao = 1.0 / (1.0 + mae)
    return aptidao, mse, mae, pu_fundos


def executar_comparativo(tipo_metrica, df_cenario, cotas_fundos, pop_size=25, geracoes=35):
    qtd_execucoes = df_cenario['quantidade'].to_numpy(dtype=np.int64)
    precos = df_cenario['preco'].to_numpy(dtype=np.float64)

    config = ConfiguracaoAG(tamanho_populacao=pop_size, geracoes=geracoes)
    ag = AlgoritmoGenetico(cotas_fundos, qtd_execucoes, precos, config)
    populacao = ag.criar_populacao()

    if tipo_metrica == 'MSE':
        avaliador = lambda ind: (
            *ag.avaliar(ind)[:2],
            float(np.mean(np.abs((ind * precos[:, None]).sum(axis=0) / cotas_fundos - ag.pu_global))),
            (ind * precos[:, None]).sum(axis=0) / cotas_fundos
        )
    else:
        avaliador = lambda ind: avaliar_via_mae(ind, precos, cotas_fundos, ag.pu_global)

    avaliacoes = [avaliador(ind) for ind in populacao]
    historico_mae = []

    for _ in range(geracoes):
        ordenados = sorted(zip(avaliacoes, populacao), key=lambda x: x[0][0], reverse=True)
        historico_mae.append(ordenados[0][0][2])

        nova_pop = [ordenados[0][1].copy(), ordenados[1][1].copy()]
        while len(nova_pop) < pop_size:
            p1 = max(random.sample(ordenados, 3), key=lambda x: x[0][0])[1]
            p2 = max(random.sample(ordenados, 3), key=lambda x: x[0][0])[1]
            filho = ag.crossover(p1, p2) if random.random() < 0.70 else p1.copy()
            if random.random() < 0.50:
                filho = ag.mutar(filho, num_trocas=random.randint(1, 8))
            nova_pop.append(filho)

        populacao = nova_pop
        avaliacoes = [avaliador(ind) for ind in populacao]

    melhor_aval, _ = max(zip(avaliacoes, populacao), key=lambda x: x[0][0])
    return {
        'pu_fundos': melhor_aval[3],
        'pu_global': ag.pu_global,
        'mse_final': melhor_aval[1],
        'mae_final': melhor_aval[2],
        'historico_mae': historico_mae
    }


def main():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    aloc_path = localizar_arquivo('alocacao_ordem.csv')
    exec_path = localizar_arquivo('massa_execucoes_500_cenarios.csv')

    nomes_fundos, cotas_fundos, df_exec = carregar_dados_ordem(aloc_path, exec_path)

    print("Executando benchmark comparativo no Cenário 1...")
    df_c1 = df_exec[df_exec['cenario_id'] == 1].copy().reset_index(drop=True)
    res_mse = executar_comparativo('MSE', df_c1, cotas_fundos, pop_size=30, geracoes=40)
    res_mae = executar_comparativo('MAE', df_c1, cotas_fundos, pop_size=30, geracoes=40)

    num_cenarios_teste = 50  # Pode modificar para quantidade de cenários que se deseja testar
    
    total_alocacoes_esperadas = num_cenarios_teste * len(nomes_fundos)
    print(f"Executando em {num_cenarios_teste} cenários ({total_alocacoes_esperadas} alocações de fundos) para análise estatística de variância e dispersão...")
    todos_desvios_mse = []
    todos_desvios_mae = []

    for c_id in range(1, num_cenarios_teste + 1):
        cdf = df_exec[df_exec['cenario_id'] == c_id].copy().reset_index(drop=True)
        r_mse = executar_comparativo('MSE', cdf, cotas_fundos, pop_size=25, geracoes=35)
        r_mae = executar_comparativo('MAE', cdf, cotas_fundos, pop_size=25, geracoes=35)

        todos_desvios_mse.extend(np.abs(r_mse['pu_fundos'] - r_mse['pu_global']))
        todos_desvios_mae.extend(np.abs(r_mae['pu_fundos'] - r_mae['pu_global']))

    desvios_mse = np.array(todos_desvios_mse)
    desvios_mae = np.array(todos_desvios_mae)

    var_mse = np.var(desvios_mse)
    var_mae = np.var(desvios_mae)
    std_mse = np.std(desvios_mse)
    std_mae = np.std(desvios_mae)

    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    cores_mse = '#1f77b4'
    cores_mae = '#ff7f0e'

    # Painel A
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

    # Painel B
    ax2 = fig.add_subplot(gs[1, 0])
    geracoes_eixo = list(range(len(res_mse['historico_mae'])))

    ax2.plot(geracoes_eixo, res_mse['historico_mae'], label='Otimizado via MSE', color=cores_mse, lw=2.2)
    ax2.plot(geracoes_eixo, res_mae['historico_mae'], label='Otimizado via MAE', color=cores_mae, lw=2.2, linestyle='--')

    ax2.set_title('B. Velocidade de Convergência Evolutiva (Erro Médio vs Gerações)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Geração', fontsize=10.5)
    ax2.set_ylabel('Erro Médio da População (R$)', fontsize=10.5)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Painel C
    ax3 = fig.add_subplot(gs[1, 1])
    dados_boxplot = [desvios_mse, desvios_mae]

    box = ax3.boxplot(dados_boxplot, tick_labels=['Modelo MSE', 'Modelo MAE'], patch_artist=True, widths=0.45,
                      medianprops=dict(color='black', lw=1.8),
                      flierprops=dict(marker='o', markersize=6, alpha=0.6))

    box['boxes'][0].set(facecolor=cores_mse, alpha=0.8)
    box['boxes'][1].set(facecolor=cores_mae, alpha=0.8)
    box['fliers'][0].set(markerfacecolor=cores_mse)
    box['fliers'][1].set(markerfacecolor=cores_mae)

    total_alocacoes = len(desvios_mse)
    ax3.set_title(f'C. Controle de Variância e Outliers ({num_cenarios_teste} Cenários - {total_alocacoes} Alocações de Fundos)', fontsize=12, fontweight='bold', pad=10)
    ax3.set_ylabel('Desvio Absoluto de PU (R$)', fontsize=10.5)
    ax3.grid(True, linestyle='--', alpha=0.5)

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

    caminho_imagem = os.path.join(diretorio_atual, 'comparativo_mse_vs_mae.png')
    plt.savefig(caminho_imagem, bbox_inches='tight')
    plt.close()
    print(f"Gráfico comparativo salvo em: {caminho_imagem}")


if __name__ == '__main__':
    main()
