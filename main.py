import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen
from nose.tools import set_trace
from statistics import mean

def init_population(pop_size, dimension):
    population = []
    for i in range(pop_size):
        individual = []
        for j in range(dimension):
            gene = random.uniform(-5.12,5.12)
            individual.append(gene)
        population.append(individual)
    return population

def crossover(current_ind, best_ind, C):
	dimension = len(current_ind)
	new_ind = [None] * dimension
	pool = []
	for i in range(dimension):
		index = np.random.randint(0,dimension)
		while index in pool:
			index = np.random.randint(0,dimension)
		pool.append(index)
		if i < C:		
			new_ind[index] = best_ind[index]
		else:
			new_ind[index] = current_ind[index] 

	return new_ind

def mutation(current_ind, t, M, MaxG):
	dimension = len(current_ind)
	mutant_ind = []
	mut_genes = []
	pool = []
	for i in range(M):
            index = np.random.randint(0,dimension)
            while index in pool:
                    index = np.random.randint(0,dimension)
            pool.append(index)
            mut_genes.append(index)	

	for j in range(dimension):
            if j in mut_genes:
                    m = current_ind[j] + (MaxG - t) * random.uniform(0, 1)
                    print(m)
            else:
                    m = current_ind[j]
            mutant_ind.append(m) 


	return mutant_ind

def get_worst(sol, population):
	worst = sol
	worst_idx = None
	for idx, ind in enumerate(population):
		if calc_obj_func(ind) >= calc_obj_func(worst):
			worst = ind
			worst_idx = idx
	return worst_idx

def get_best(population):
	best = population[0]
	for ind in population:
		if calc_obj_func(ind) < calc_obj_func(best):
			best = ind
	return best

def calc_obj_func(sol, obj_func='sphere'):
	result = 0	
	if obj_func == 'sphere':
		result = sphere(sol)
	elif obj_func == 'rosenbrock':
		result = rosen(sol)
	else:
		return 0
	return result


def generation(t,parameters,population,costs,bests):

	best = get_best(population)
	bests.append(calc_obj_func(best))

	# çaprazlama
	for ind in population:
		if random.uniform(0, 1) < parameters["CR"]:
			candidate = crossover(ind,best,parameters["C"])

			worst_idx = get_worst(candidate, population)			
			if worst_idx is not None:
				population[worst_idx] = candidate
				costs[worst_idx] = calc_obj_func(candidate)


	# mutasyon
	for idx, ind in enumerate(population):
            if random.uniform(0, 1) < parameters["MR"]:
                mutant = mutation(ind, t, parameters["M"], parameters["MaxG"])
                mutant_cost = calc_obj_func(mutant)
                if mutant_cost < calc_obj_func(ind):
                    population[idx] = mutant
                    costs[idx] = mutant_cost
            return population, costs

def plot(bests):
	generations = [i for i in range(len(bests))]
	plt.plot(generations,bests,'ro-')
	# zip joins x and y coordinates in pairs
	for x,y in zip(generations,bests):

	    label = "{:.2f}".format(y)

	    plt.annotate(label, # this is the text
	                 (x,y), # this is the point to label
	                 textcoords="offset points", # how to position the text
	                 xytext=(0,10), # distance from text to points (x,y)
	                 ha='center') # horizontal alignment can be left, right or center


	plt.xticks(np.arange(min(generations), max(generations)+1, 1.0))

	plt.ylabel('bests')
	plt.show()


def run():
	# Control Parameters
	parameters={}
	parameters["NP"] = 100
	parameters["D"] = 3
	parameters["MaxG"] = 50
	parameters["CR"] = 0.5
	parameters["MR"] = 0.1
	parameters["C"] = 1*parameters["D"] // 3
	parameters["M"] = 6*parameters["D"] // 6
	costs = []
	bests = []

	# başlangıç pop oluştur
	population = init_population(parameters["NP"], parameters["D"])
	
	#değerlendir
	for ind in population:
		costs.append(calc_obj_func(ind))

	for t in range(parameters["MaxG"]):
		population, costs = generation(t,parameters,population,costs, bests)
	best = calc_obj_func(get_best(population))
	bests.append(best)
	plot(bests)
	return best 

def main():
	runtime = 1 
	bests = []
	for i in range(runtime):
		bests.append(run())
	result = mean(bests)
	print(result)

def sphere(sol):
	top = 0
	for j in sol:
		top=top+j*j
	return top


main()
