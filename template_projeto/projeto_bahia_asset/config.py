from dataclasses import dataclass


@dataclass
class ConfiguracaoAG:
    """Parâmetros de controle da evolução do Algoritmo Genético."""
    tamanho_populacao: int = 25
    geracoes: int = 35
    taxa_crossover: float = 0.70
    taxa_mutacao: float = 0.50
    tamanho_elite: int = 2
    tamanho_torneio: int = 3
