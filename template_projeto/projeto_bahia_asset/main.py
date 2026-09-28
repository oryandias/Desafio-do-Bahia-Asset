import os
import sys
import time
from typing import Optional
import numpy as np
import pandas as pd

# Permite execução direta como script (python main.py) ou como submódulo de pacote
if __package__ is None or __package__ == "":
    diretorio_pai = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if diretorio_pai not in sys.path:
        sys.path.insert(0, diretorio_pai)
    from projeto_bahia_asset.config import ConfiguracaoAG
    from projeto_bahia_asset.dados import carregar_dados_ordem, salvar_resultado_excel, localizar_arquivo
    from projeto_bahia_asset.algoritmo_genetico import AlgoritmoGenetico
else:
    from .config import ConfiguracaoAG
    from .dados import carregar_dados_ordem, salvar_resultado_excel, localizar_arquivo
    from .algoritmo_genetico import AlgoritmoGenetico


def executar(
    caminho_alocacao: str,
    caminho_execucoes: str,
    caminho_saida_excel: str,
    max_cenarios: Optional[int] = None,
    configuracao: Optional[ConfiguracaoAG] = None
) -> None:
    """
    Orquestra a leitura dos dados, execução do AG para todos os cenários e exportação para Excel.
    """
    # Carrega a parametrização inicial da ordem:
    nomes_fundos, cotas_fundos, df_execucoes = carregar_dados_ordem(
        caminho_alocacao, caminho_execucoes
    )
    
    # Extrai e ordena os IDs únicos de cenários (de 1 a 500) para processamento sequencial
    ids_cenarios = sorted(df_execucoes['cenario_id'].unique())
    if max_cenarios is not None:
        ids_cenarios = ids_cenarios[:max_cenarios]

    print(f"Iniciando otimização de {len(ids_cenarios)} cenários via Algoritmo Genético...")
    tempo_inicio = time.time()

    tabelas_alocadas = []  # Armazena os DataFrames de cada cenário já com as colunas dos fundos preenchidas
    metricas = []          # Registra o histórico estatístico de cada cenário para governança e auditoria

    for idx, c_id in enumerate(ids_cenarios, 1):
        t0 = time.time()

        # Isola as execuções de mercado pertencentes exclusivamente ao cenário atual (c_id)
        sub_df = df_execucoes[df_execucoes['cenario_id'] == c_id].copy().reset_index(drop=True)

        # Converte as colunas do pandas em arrays NumPy contíguos em memória para aceleração computacional
        qtd_execucoes = sub_df['quantidade'].to_numpy(dtype=np.int64)
        precos = sub_df['preco'].to_numpy(dtype=np.float64)

        # Instancia o Algoritmo Genético para o cenário específico 
        otimizador = AlgoritmoGenetico(cotas_fundos, qtd_execucoes, precos, configuracao)
        
        # Executa o processo evolutivo:
        melhor_matriz, mse, aptidao, pu_fundos = otimizador.executar()

        # Prepara a tabela de saída do cenário: preserva os dados originais da boleta e anexa as colunas dos fundos
        df_resultado = sub_df[[
            'cenario_id', 'execucao_id', 'quantidade', 'preco', 'tipo_distribuicao'
        ]].copy()
        for i, nome in enumerate(nomes_fundos):
            # melhor_matriz[:, i] seleciona a fatia vertical (coluna) correspondente ao fundo i
            df_resultado[nome] = melhor_matriz[:, i]

        tabelas_alocadas.append(df_resultado)
        tempo_cenario = time.time() - t0

        # Mensuração fiduciária: avalia a pior discrepância pontual de preço sofrida por qualquer fundo
        desvio_maximo = float(np.max(np.abs(pu_fundos - otimizador.pu_global)))
        
        # Salva as métricas de auditoria que comporão a segunda aba do Excel
        metricas.append({
            'cenario_id': c_id,
            'mse': mse,
            'fitness': aptidao,
            'pu_global': otimizador.pu_global,
            'max_desvio_fundo': desvio_maximo,
            'tempo_segundos': tempo_cenario
        })

        if idx % 50 == 0 or idx == len(ids_cenarios) or idx <= 5:
            print(f"[{idx:03d}/{len(ids_cenarios):03d}] Cenário {c_id:03d} | MSE: {mse:.2e} | Max Desvio: R$ {desvio_maximo:.4f} | Tempo: {tempo_cenario:.2f}s")

    tempo_total = time.time() - tempo_inicio
    print(f"\nOtimização concluída em {tempo_total:.2f}s ({tempo_total / 60:.2f} min).")
    print(f"Gravando planilha de resultados: {caminho_saida_excel}")

    # Concatena os 500 DataFrames individuais em uma base única consolidada de 250.000 linhas
    df_consolidado = pd.concat(tabelas_alocadas, ignore_index=True)
    df_metricas = pd.DataFrame(metricas)

    # Grava a pasta de trabalho com as duas abas estruturadas (Divisao_Execucoes e Metricas_Cenarios)
    salvar_resultado_excel(caminho_saida_excel, df_consolidado, df_metricas)

    print("Resultados salvos com sucesso!")
    print(f"MSE Médio Geral: {df_metricas['mse'].mean():.2e}")
    print(f"Desvio Máximo Médio por Fundo: R$ {df_metricas['max_desvio_fundo'].mean():.4f}")


def main() -> None:
    """Ponto de entrada para execução direta do módulo orquestrador."""
    diretorio_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_aloc = localizar_arquivo('alocacao_ordem.csv')
    caminho_exec = localizar_arquivo('massa_execucoes_500_cenarios.csv')
    caminho_excel = os.path.join(diretorio_base, 'resultado_divisao.xlsx')
    executar(caminho_aloc, caminho_exec, caminho_excel)


if __name__ == '__main__':
    main()
