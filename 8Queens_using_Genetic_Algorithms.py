import numpy as np
import math
from collections import defaultdict

class NQueensGenetic:
  def __init__(self , mutation_rate , n, init_size):
    self.mutation_rate = mutation_rate
    self.board = [[0] * n for __ in range(n)]
    self.init_size = init_size
    self.population = []
    self.n = n
    self.maxFitness = (n * (n - 1)) / 2
    self.initPopulation()

  def initPopulation(self):
    for iter in range(self.init_size):
      r = np.arange(0 , self.n)
      np.random.shuffle(r)
      self.population.append(list(r))

  def printBoard(self):
    for i in range(self.n):
      for j in range(self.n):
        print("Q" if self.board[i][j] else "X", end= ' ' if j + 1 < self.n else '\n')
  
  def resetBoard(self):
    self.board = [[0] * self.n for __ in range(self.n)]

  def populate(self , chromosome):
    for i in range(len(chromosome)):
      self.board[chromosome[i]][i] = 1
  
  def crossover(self ,a , b):
    l = np.random.randint(1 , len(a))
    r = np.random.randint(l , len(a))
    a = list(a)
    b = list(b)
    middle = a[l : r + 1]
    edges = []
    for x in b:
      if x not in middle:
        edges.append(x)
    
    m = 0
    child = a
    for i in range(len(a)):
      if i  not in range(l , r + 1):
        child[i] = edges[m]
        m += 1
    return child

  def mutate(self, chromosome):
    #randomly select a pair of alleles and swap them
    pairs = int(self.mutation_rate * len(self.population))
    swap_pairs = np.random.choice(a = chromosome , size = (int(pairs / 2) , 2) , replace = False)
    for a , b in swap_pairs:
      chromosome[a] , chromosome[b] = chromosome[b] , chromosome[a]
    return chromosome

  def fitness(self, chromosome):
    self.resetBoard()
    self.populate(chromosome)
    left_diagonal = defaultdict(int)
    right_diagonal = defaultdict(int)
    row = defaultdict(int)
    col = defaultdict(int)

    rowCollisions = 0
    diagonalCollisions = 0
    for i  in range(self.n):
      for j in range(self.n):
        row[i] += self.board[i][j]
        col[j] += self.board[i][j]
        right_diagonal[i - j] += self.board[i][j]
        left_diagonal[i + j]  += self.board[i][j]
    
    for key , count in left_diagonal.items():
      diagonalCollisions += max(0 , count - 1)

    for key, count in right_diagonal.items():
      diagonalCollisions += max(0 ,count - 1)
    
    for key, count in row.items():
      rowCollisions += max(0 ,count - 1)
    
    for key, count in col.items():
      rowCollisions += max(0 , count - 1)
    
    netCollisions = rowCollisions + diagonalCollisions
    return int(self.maxFitness - netCollisions)
  
  def fitnessProbability(self , chromosome):
    return self.fitness(chromosome) / sum([self.fitness(x) for x in self.population])

  def simulate(self):
    currentBest = 0
    iterations = 0
    max_iterations = 5000

    while currentBest < self.maxFitness and iterations < max_iterations:
      newPopulation = []
      for i in range(len(self.population)):
        k ,l  = np.random.choice(len(self.population) , 2, replace = False)
        x = self.population[k]
        y = self.population[l]
        child = self.crossover(x , y)
        child = self.mutate(child)
        print('New chromosome : ' , child, ' Fitness : ', self.fitness(child))
        newPopulation.append(child)
        if(self.fitness(child) == self.maxFitness):
          break

      self.population = newPopulation
      currentBest = max(currentBest , max([self.fitness(x) for x in self.population]))
      
      print('\nBest fitness in current generation: ' , max([self.fitness(x) for x in self.population]))
      print('Maximum fitness seen after generation {gen} is '.format(gen = iterations + 1) , currentBest, end = '\n')

      if(currentBest == self.maxFitness):
        for x in self.population:
          if self.fitness(x) == self.maxFitness:
            print('\nFound valid solution after {} generations !'.format(iterations + 1))
            print('\nOne of the solutions for 8 queens problem is given by :' , x, 'with fitness ' , self.fitness(x))
            return x
      iterations += 1

    return []
  

seen = set()
for i in range(92):
  EightQueens = NQueensGenetic(mutation_rate=0.3 , n = 8, init_size = 15)
  s = EightQueens.simulate()
  if(s is not None):
    seen.add(tuple(s))
  if(len(seen) >= 7):
    break

print('\nSeven different solutions for 8 queens problem are: ' , end = '\n')

i = 1
for solution in seen:
  print('Solution {} : '.format(i))
  print(list(solution))
  solutionObject = NQueensGenetic(mutation_rate = 0.3 , n = 8, init_size = 15)
  solutionObject.populate(solution)
  solutionObject.printBoard()
  print("-------------------------------xxxxx-----------------------------")
  print()
  i += 1

