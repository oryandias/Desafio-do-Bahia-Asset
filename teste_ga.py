import random

def criar_individuo():
    # Representação fictícia: um vetor de escolhas de alocação
    return [random.randint(0, 100) for _ in range(10)]

def calcular_fitness(individuo):
    # Função objetivo: queremos minimizar o erro (retornamos negativo ou inverso para maximizar)
    erro = sum(abs(x - 50) for x in individuo) # Exemplo: meta é 50
    return 1.0 / (1.0 + erro)

def selecao(populacao):
    # Seleção por torneio simples
    pais = []
    for _ in range(2):
        candidatos = random.sample(populacao, 3)
        melhor = max(candidatos, key=calcular_fitness)
        pais.append(melhor)
    return pais

def crossover(pai1, pai2):
    ponto = len(pai1) // 2
    filho1 = pai1[:ponto] + pai2[ponto:]
    filho2 = pai2[:ponto] + pai1[ponto:]
    return filho1, filho2

def mutacao(individuo, taxa=0.1):
    for i in range(len(individuo)):
        if random.random() < taxa:
            individuo[i] = random.randint(0, 100)
    return individuo

# --- Loop Principal do AG ---
tamanho_populacao = 50
geracoes = 100

populacao = [criar_individuo() for _ in range(tamanho_populacao)]

for g in range(geracoes):
    nova_populacao = []
    while len(nova_populacao) < tamanho_populacao:
        pai1, pai2 = selecao(populacao)
        filho1, filho2 = crossover(pai1, pai2)
        nova_populacao.append(mutacao(filho1))
        nova_populacao.append(mutacao(filho2))
    populacao = nova_populacao

melhor_solucao = max(populacao, key=calcular_fitness)
print("Melhor solução encontrada:", melhor_solucao)