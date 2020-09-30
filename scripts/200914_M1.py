#!/usr/bin/env python

# Author: Kimiko Suzuki
# Date: 181013
# Notes: python35

# 200817 -> SSE

###################################################################
#IMPORT PACKAGES
###################################################################
import numpy as np
from scipy.integrate import odeint
# from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from deap import base, creator, tools, algorithms
import os
import sys
import pickle
import time as timeski
import math
# from itertools import product
import pathlib
import pandas as pd
# import multiprocessing
# import sys
# import pathlib

sys.path.insert(1, '../python_modules/')
import model
import model_supp

###################################################################
#MATRIX FOR VARIABLES TO INTERP AND EXPONENTIATE
###################################################################



#############################################################################
#PARAMETERS
#############################################################################


#############################################################################
#EA FUNCTIONS
#############################################################################



def scorefxn_helper(individual):
    # just a helper function that pulls all of scorefxn1 dependencies together
    # note the (), <--using single optimization in DEAP for now
    # scorefxn1 is taking care of the multiple optimizations for now
    return model_supp.calc_sim_score(model_fxns, exp_data, exp_time, total_protein, inits, individual, ptpD=False, convert=(True, arr_conversion_matrix)),



###################################################################
#CHECK FOR / CREATE DIR FOR DATA
###################################################################
def strip_filename(fn):
    #input = full path filename
    #output = filename only
    #EX input = '/home/iammoresentient/phd_lab/data/data_posnegfb_3cellsum.pickled'
    #EX output = 'data_posnegfb_3cellsum'
    if '/' in fn:
        fn = fn.split('/')[-1]
    fn = fn.split('.')[0]
    return fn


def add_info(fn, gens, inds, mutationrate, crossoverrate):
    #input = filename only
    #output = date + filename + EA data
    # EX input = 'data_posnegfb_3cellsum'
    # EX output = '170327_data_posnegfb_3cellsum_#g#i#m#c'

    #get current date:
    cur_date = timeski.strftime('%y%m%d')
    # setup EA data:
    ea_data = str(gens) + 'g' + str(inds) + 'i' + str(int(mutationrate*100)) + 'm' + str(int(crossoverrate*100)) + 'c'
    #put it all together:
    #new_fn = cur_date + '_' + fn + '_' + ea_data
    new_fn = cur_date + '_' + os.path.basename(fn).split('.')[0].split('_')[-1] + '_' + ea_data
    return new_fn




###################################################################
#LOOP: EVOLUTIONARY ALGORITHM + SAVE DATA
##################################################################
def run(): # DOGWOOD

    stripped_name = strip_filename(save_filename)
    informed_name = add_info(stripped_name, number_of_generations, number_of_individuals, mutation_rate, crossover_rate)
    fn_to_use = informed_name
    dir_to_use = os.getcwd() + '/' + stripped_name

    #check if dir exists and make
    if not os.path.isdir(dir_to_use):
        os.makedirs(dir_to_use)
        # print('Made: ' + dir_to_use)
        # and make README file:
        fn = dir_to_use + '/' + 'output.txt'
        file = open(fn, 'w')

        # write pertinent info at top
        file.write('OUTPUT\n\n')
        file.write('Filename: ' + stripped_name + '\n')
        file.write('Directory: ' + dir_to_use + '\n')
        file.write('Data file: ' + save_filename + '\n\n')
        file.write('Generations: ' + str(number_of_generations) + '\n')
        file.write('Individuals: ' + str(number_of_individuals) + '\n')
        file.write('Mutation rate: ' + str(mutation_rate) + '\n')
        file.write('Crossover rate: ' + str(crossover_rate) + '\n')
        file.write('\n\n\n\n')

        #write script to file
        #script_name = os.getcwd() + '/' + 'EA_1nf1pf.py'
        script_name = os.path.basename(__file__)#__file__)
        open_script = open(script_name, 'r')
        write_script = open_script.read()
        file.write(write_script)
        open_script.close()

        file.close()

    for i in range(number_of_runs):
        ###################################################################
        #EVOLUTIONARY ALGORITHM
        ###################################################################
        #TYPE
        #Create minimizing fitness class w/ single objective:
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        #Create individual class:
        creator.create('Individual', list, fitness=creator.FitnessMin)

        #TOOLBOX
        toolbox = base.Toolbox()
        #Register function to create a number in the interval [1-100?]:
        #toolbox.register('init_params', )
        #Register function to use initRepeat to fill individual w/ n calls to rand_num:
        toolbox.register('individual', tools.initRepeat, creator.Individual,
                         np.random.random, n=number_of_params)
        #Register function to use initRepeat to fill population with individuals:
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        #GENETIC OPERATORS:
        # Register evaluate fxn = evaluation function, individual to evaluate given later
        toolbox.register('evaluate', scorefxn_helper)
        # Register mate fxn = two points crossover function
        toolbox.register('mate', tools.cxTwoPoint)
        # Register mutate by swapping two points of the individual:
        toolbox.register('mutate', tools.mutPolynomialBounded,
                         eta=0.1, low=0.0, up=1.0, indpb=0.2)
        # Register select = size of tournament set to 3
        toolbox.register('select', tools.selTournament, tournsize=3)

        #EVOLUTION!
        pop = toolbox.population(n=number_of_individuals)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(key = lambda ind: [ind.fitness.values, ind])
        stats.register('all', np.copy)

        # using built in eaSimple algo
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=crossover_rate,
                                           mutpb=mutation_rate,
                                           ngen=number_of_generations,
                                           stats=stats, halloffame=hof,
                                           verbose=False)
        # print(f'Run number completed: {i}')

        ###################################################################
        #MAKE LISTS
        ###################################################################
        # Find best scores and individuals in population
        arr_best_score = []
        arr_best_ind = []
        for a in range(len(logbook)):
            scores = []
            for b in range(len(logbook[a]['all'])):
                scores.append(logbook[a]['all'][b][0][0])
            #print(a, np.nanmin(scores), np.nanargmin(scores))
            arr_best_score.append(np.nanmin(scores))
            #logbook is of type 'deap.creator.Individual' and must be loaded later
            #don't want to have to load it to view data everytime, thus numpy
            ind_np = np.asarray(logbook[a]['all'][np.nanargmin(scores)][1])
            ind_np_conv = model_supp.convert_individual(ind_np, arr_conversion_matrix, number_of_params)
            arr_best_ind.append(ind_np_conv)
            #arr_best_ind.append(np.asarray(logbook[a]['all'][np.nanargmin(scores)][1]))


        # print('Best individual is:\n %s\nwith fitness: %s' %(arr_best_ind[-1],arr_best_score[-1]))

        ###################################################################
        #PICKLE
        ###################################################################
        arr_to_pickle = [arr_best_score, arr_best_ind]

        def get_filename(val):
            filename_base = dir_to_use + '/' + stripped_name + '_'
            if val < 10:
                toret = '000' + str(val)
            elif 10 <= val < 100:
                toret = '00' + str(val)
            elif 100 <= val < 1000:
                toret = '0' + str(val)
            else:
                toret = str(val)
            return filename_base + toret + '.pickled'

        counter = 0
        filename = get_filename(counter)
        while os.path.isfile(filename) == True:
            counter += 1
            filename = get_filename(counter)

        pickle.dump(arr_to_pickle, open(filename,'wb'))
    #     print('Dumped data to file here: ', filename)

# FOR DOGWOOD
# def throw_away_function(_):
#     return run()
#
# def main():
#     pool=multiprocessing.Pool(processes=number_of_runs)
#     pool.map(throw_away_function, range(number_of_runs))
#
if __name__ == '__main__':
    # define where to save runs
    save_filename = '200914_M1.txt'

    # define experimental data
    exp_data, exp_time = model_supp.get_data(local=False)

    # define total protein concentrations
    MAP3K_t = model_supp.molarity_conversion(123+1207+1611) #ssk2+ssk22+ste11
    MAP2K_t = model_supp.molarity_conversion(4076)
    MAPK_t = model_supp.molarity_conversion(8225)
    PTP_t = model_supp.molarity_conversion(443+1324) # ptp2+ptp3
    total_protein = [MAP3K_t, MAP2K_t, MAPK_t, 1] #uM, except for gly (1) which is a placeholder for multiplying arrays together

    #  define initial conditions
    MAP3K = 0.05*MAP3K_t # estimated (so not 0)
    MAP2K = 0.05975380333*MAP2K_t # from the biological data
    MAPK = 0.00540042381*MAPK_t  # from the biological data
    gly = 0.00001 # placeholder (so not 0)
    PTP = model_supp.molarity_conversion(443+1324) # ptp2+ptp3
    inits = [MAP3K, MAP2K, MAPK]

    # doses defined in model_supp.calc_sim_score()
    # hog1_doses = [0, 50000, 150000, 250000, 350000, 450000, 550000]
    # hog1_t100a_doses = [0, 50000, 150000, 250000, 350000, 450000, 550000]
    # pbs2_doses = [150000, 550000]
    # ptp_doses = [0, 150000, 550000]

    # EA params
    number_of_runs = 5
    number_of_generations = 500
    number_of_individuals = 500
    mutation_rate = 0.1
    crossover_rate = 0.5
    number_of_params = 14

    # EA param ranges
    minimums = [-4, -4,
        -4, -4, -4, -4,
        -4, -4, -4, -4, -4,
        -4, -4, -4]

    maximums = [ 4, 4,
        4, 4, 4, 4,
        4, 4, 4, 4, 4,
        4, 4, 4]

    #conversion matrix
    arr_conversion_matrix = model_supp.make_conversion_matrix(number_of_params, maximums, minimums)


    # define model functions
    model_fxns = model.Model(model.M1_kb, model.simulate_t100a_experiment_M1_kb)

    run()
