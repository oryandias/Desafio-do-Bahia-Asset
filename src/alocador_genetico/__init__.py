"""
Pacote alocador_genetico
Módulos do Algoritmo Genético para divisão justa de execuções entre fundos.
"""

from .config import ConfiguracaoAG
from .dados import carregar_dados_ordem, salvar_resultado_excel, localizar_arquivo
from .algoritmo_genetico import AlgoritmoGenetico
from .main import executar

__all__ = [
    'ConfiguracaoAG',
    'carregar_dados_ordem',
    'salvar_resultado_excel',
    'localizar_arquivo',
    'AlgoritmoGenetico',
    'executar',
]
