"""
Ponto de entrada oficial para a divisão de execuções entre fundos de investimento.
Desafio Técnico: Bahia Asset Management
"""

import os
import sys

# Garante a resolução do pacote local independente do diretório de chamada
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alocador_genetico.main import main

if __name__ == '__main__':
    main()
