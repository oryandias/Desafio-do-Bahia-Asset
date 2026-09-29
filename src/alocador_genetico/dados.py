import os
import pandas as pd
import numpy as np
from typing import Tuple, List


def localizar_arquivo(nome_arquivo: str) -> str:
    """Busca o arquivo no diretório de execução atual, no diretório local ou nos diretórios pais."""
    caminhos_tentativa = [
        nome_arquivo,
        os.path.join(os.getcwd(), nome_arquivo),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', nome_arquivo),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', nome_arquivo),
    ]
    for c in caminhos_tentativa:
        if os.path.exists(c):
            return os.path.abspath(c)
    return nome_arquivo


def carregar_dados_ordem(
    caminho_alocacao: str, 
    caminho_execucoes: str
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    """Lê os arquivos CSV com as cotas contratadas pelos fundos e as execuções do mercado."""
    df_alocacao = pd.read_csv(localizar_arquivo(caminho_alocacao))
    df_execucoes = pd.read_csv(localizar_arquivo(caminho_execucoes))

    nomes_fundos = df_alocacao['fundo'].tolist()
    cotas_fundos = df_alocacao['quantidade'].to_numpy(dtype=np.int64)

    return nomes_fundos, cotas_fundos, df_execucoes


def salvar_resultado_excel(
    caminho_saida: str,
    df_alocacoes_consolidadas: pd.DataFrame,
    df_metricas_cenarios: pd.DataFrame
) -> None:
    """Grava o resultado final na planilha Excel com as abas de divisão e métricas consolidadas."""
    with pd.ExcelWriter(caminho_saida, engine='openpyxl') as writer:
        df_alocacoes_consolidadas.to_excel(
            writer, sheet_name='Divisao_Execucoes', index=False
        )
        df_metricas_cenarios.to_excel(
            writer, sheet_name='Metricas_Cenarios', index=False
        )
