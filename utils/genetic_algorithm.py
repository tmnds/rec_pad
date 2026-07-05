import numpy as np
import random

from sklearn.metrics import mean_squared_error

class GeneticAlgorithm:
    def __init__(
            self, 
            max_size_population=20, 
            max_genes=4,
            generations=50, 
            size_tournament=3
        ):

        self.max_size_population = max_size_population
        self.max_genes = max_genes
        self.generations = generations
        self.size_tournament = size_tournament

        self.best_all_fits = []
        self.global_best_fits = -np.inf
        self.best_ind = None

        # self.rng = np.random.default_rng() Caso seja necessário

    def start_population(self):
        '''
        Inicialização
        Criação aleatória de um conjunto de soluções candidatas (população inicial).
        **Aqui vamos criar individuos de forma aleatória para cada um dos modelos que serão utilizados.**

        20 Individuos com espaco de busca de 3 dimensões (3 modelos), 
        de pesos continuos, com valores entre 0 e 1, e a soma dos 
        pesos deve ser igual a 1.

        # # Opção usando NumPy - dirichlet
        #       gene = np.random.dirichlet(
        #           alpha=np.ones(4),
        #           size=1
        #       )
        
        '''
        population = []
        epsilon = 1e-6

        for _ in range(self.max_size_population):
            
            genes = np.array(
                [
                    random.uniform(epsilon, 1)
                    for _ in range(self.max_genes)
                ]
            )

            genes = genes / np.sum(genes)
            random.shuffle(genes)
            population.append(genes)
            # self.population.append(genes)

        return np.array(population)

    def evaluate_population(self, y_test, y_hat):
        '''
        2. Avaliação
        
        Aplica-se uma função de fitness no conjunto.
        Essa função define o quão boa é cada solução (quanto maior ou menor, melhor — depende do problema).
        Nesse processo, os individuos inicializados serão executados em cada modelo e avaliados de acordo com o resultado do fitness utilizado.
        
        '''
        ind_metrics = []
        
        for _ in range(y_hat.shape[1]):
            fitness = 1 / mean_squared_error(y_test, y_hat[:,_])
            ind_metrics.append(fitness)

        return ind_metrics

    def tournament_selection(self, population, fitness, size_tournament):
        '''
        Seleção
        Alguns dos melhores candidatos são escolhidos.
        Esses candidatos irão gerar a próxima geração.

        Processo probabilístico, onde indivíduos com melhor fitness têm maior chance de serem selecionados, mas mesmo os piores têm uma chance (evitando convergência prematura).
        
        '''
        n_population = len( population )
        selected = []

        for _ in range(n_population):
            participants = random.sample( 
                range( len(population) ), size_tournament 
            )

            winner = participants[0]
            for i in range(1, size_tournament):
                if fitness[participants[i]] > fitness[winner]:
                    winner = participants[i]
            
            selected.append(population[winner])

        return selected


    def crossover(self, selected_parents):
        '''
        Cruzamento (Recombinação)
        
        - A partir de dois ou mais candidatos (pais), são gerados novos indivíduos (filhos). 
        '''
        new_generation = []

        for gen in range(0, len(selected_parents), 2):
            parent_1, parent_2 = selected_parents[gen], selected_parents[gen+1]
            single_point = random.randint(1, len(parent_1) - 1)

            child_1 = np.concatenate((parent_1[:single_point], parent_2[single_point:]))
            child_2 = np.concatenate((parent_2[:single_point], parent_1[single_point:]))
            
            new_generation.extend([child_1, child_2])
        
        return new_generation

    def mutation(self, offspring):
        '''
        Mutação
        - Pequenas alterações aleatórias são introduzidas em alguns indivíduos.
        - Isso ajuda a manter a diversidade genética e evita que o algoritmo fique preso em ótimos locais.

        - Se usarmos child[gene] = random.uniform(0, 1) - Dessa forma eu mudo completamente o Gene, o que teoricamente favorece Exploração, comportamentoo mais agfressivo, diversidade e menos estabilidade

        - Se usarmos child[gene] += random.uniform(-0.001, 0.001) - dessa forma eu favoreço refinamento local, busca gradual, comportamento mais estável, e exploração fina.

        '''
        mutation_rate = 0.1

        for i, child in enumerate(offspring):
            mutation = random.random()

            if mutation < mutation_rate:

                gene = random.randint(0, len(child) - 1)
                child[gene] += random.uniform(-0.001, 0.001)

            child = np.clip(child, 0, None)
            child = child / np.sum(child) # Renormalização para garantir que a soma dos pesos seja igual a 1
            
            offspring[i] = child
        
        return np.array(offspring)

    def execute(self, P, y_test):
        '''
        Método responsável pela Execução do GA.

        '''

        W = self.start_population()

        for _ in range(self.generations):
            y_hat = P @ W.T
            fitness = self.evaluate_population(y_test, y_hat)
            self.best_all_fits.append(max(fitness))

            idx = np.argmax(fitness)

            if fitness[idx] > self.global_best_fits:
                self.global_best_fits = fitness[idx]
                self.best_ind = W[idx].copy()

            selected_parents = self.tournament_selection(
                W, 
                fitness, 
                self.size_tournament
            )
            new_generation = self.crossover(selected_parents)
            W = self.mutation(new_generation)
        
        return {
            'best_ind': self.best_ind,
            'best_fitness': self.global_best_fits,
            'best_mse': 1 / self.global_best_fits,
            'fitness_curve': self.best_all_fits
        }
 