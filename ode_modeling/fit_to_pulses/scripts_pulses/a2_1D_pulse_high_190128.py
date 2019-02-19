#!/usr/bin/env python

# Author: Kimiko Suzuki
# Date: 181013
# Notes: python35

###################################################################
#IMPORT PACKAGES
###################################################################
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from deap import base, creator, tools, algorithms
import os
import sys
import pickle
import time as timeski
import math
from itertools import product
import pathlib
import pandas as pd
import multiprocessing
# import sys
# import pathlib

###########################################################################
#LOAD EXPERIMENTAL DATA
###########################################################################

save_filename = '190128_a2_high_pulse.txt'

wt_folder = '/nas/longleaf/home/sksuzuki/HOG_model/data_high/MAPK activation/WT'
# t100a_folder = '/nas/longleaf/home/sksuzuki/HOG_model/data_pbs2/MAPK activation/T100A'
# pbs2_folder = '/nas/longleaf/home/sksuzuki/HOG_model/data_pbs2/MAPK activation/Pbs2'
# pbs2_t100a_folder = '/nas/longleaf/home/sksuzuki/HOG_model/data_pbs2/MAPK activation/Pbs2_T100A'
mapk_pulse = '/nas/longleaf/home/sksuzuki/HOG_model/data_pbs2/MAPK activation/pulse_hog1'
# map2k_pulse = '/nas/longleaf/home/sksuzuki/HOG_model/data_pbs2/MAPK activation/pulse_pbs2'


def load_csv_data(folder):
    data = []
    for csv in pathlib.Path(folder).glob('*.csv'):
        f_data = pd.read_csv(csv)
        time = f_data['Time'].tolist()
        f_data=f_data.set_index('Time')
        f_data = f_data.mean(axis=1)
        f_data = f_data.tolist()
        data.append(f_data)
    return time, data

mapk_time, mapk_wt_data = load_csv_data(wt_folder)
# mapk_time, mapk_t100a_data = load_csv_data(t100a_folder)
# mapk_time, map2k_wt_data = load_csv_data(pbs2_folder)
# mapk_time, map2k_t100a_data = load_csv_data(pbs2_t100a_folder)
mapk_pulse_time, mapk_pulse_data = load_csv_data(mapk_pulse)
# mapk_pulse_time, map2k_pulse_data = load_csv_data(map2k_pulse)

###################################################################
#EA PARAMS
###################################################################

number_of_runs = 200
number_of_generations = 250
number_of_individuals = 500
mutation_rate = 0.1
crossover_rate = 0.5
number_of_params = 16

#############################################################################
#Convert molecules to molar concentration
#############################################################################
def molarity_conversion(molecules):
    Na = 6.02214076*10**23
    cell_volume = 44                                 # volume of a yeast cell
    return molecules/(Na*cell_volume*10**-15)*1000000 # returns uM

# Protein concentrations (uM)
MAP3K_t = molarity_conversion(704)
MAP2K_t = molarity_conversion(2282)
MAPK_t = molarity_conversion(5984)

###################################################################
#MATRIX FOR VARIABLES TO INTERP AND EXPONENTIATE
###################################################################

def make_conversion_matrix(number_of_params):
    # want easily savable matrix to hold this info
    # interp boolean, interp range (min,max), power boolean, power number (y)
    arr_IandP = np.zeros((5,number_of_params))
    # Set all interp booleans to 1 - everything is going to be interpreted
    arr_IandP[0,:] = 1
    # Set all power booleans to 1 - everything is in the form of powers
    arr_IandP[3,:] = 1
    # Set all power numbers to 10 - everything has a base of 10
    arr_IandP[4,:] = 10
    # Set minimums and maximums for all parameters. Parameters are in the following order:
    # beta_0,    k1, k3, k5, k7, k9,    k2, k4, k6, k8, k10,       K_1, K_3, K_5, K_7, K_9,    K_2, K_4, K_6, K_8, K_10
    minimums = [-8, -4,
        -4, -4, -4, -4,
        -4, -4, -4, -4,
        -4, -4, -4,
        -4, -4, -4]

    maximums = [ 2, 4,
        4, 4, 4, 4,
        4, 4, 4, 4,
        4, 4, 4,
        4, 4, 4]

    for i in range(len(minimums)):
        arr_IandP[1,i] = minimums[i] #interp_range_min
        arr_IandP[2,i] = maximums[i] #interp_range_max

    return arr_IandP


#############################################################################
#PARAMETERS
#############################################################################
# initial values
MAP3K = 0
MAP2K = 0
MAPK = 0
X = 0

# signal strengths (uM)

experiment_data = {
                    # 0 : [mapk_wt_data[0], mapk_t100a_data[0]],
                    # 50000 : [mapk_wt_data[1], mapk_t100a_data[1]],
                    # 150000 : [mapk_wt_data[2], mapk_t100a_data[2], map2k_wt_data[0], map2k_t100a_data[0]],
                    # 250000 : [mapk_wt_data[3], mapk_t100a_data[3]],
                    350000 : [mapk_wt_data[0]],#, mapk_t100a_data[4]],
                    450000 : [mapk_wt_data[1]],#, mapk_t100a_data[5]],
                    550000 : [mapk_wt_data[2], mapk_pulse_data]#, map2k_pulse_data]
    }

experiment_time = {
                    # 0 : [mapk_time],
                    # 50000 : [mapk_time],
                    # 150000 : [mapk_time],
                    # 250000 : [mapk_time],
                    350000 : [mapk_time],
                    450000 : [mapk_time],
                    550000 : [mapk_time, mapk_pulse_time]
    }
experiment_doses = [350000, 450000, 550000]

signal_params = 10 # len_period

inits = [MAP3K, MAP2K, MAPK, X]
total_protein = [MAP3K_t, MAP2K_t, MAPK_t] #uM

#conversion matrix
arr_conversion_matrix = make_conversion_matrix(number_of_params)
#############################################################################
# MODEL
#############################################################################


def b3_a2_1D_X(initals,t,total_protein,sig,params, signal_fxn=None, signal_params=None):
    MAP3K, MAP2K, MAPK, X = initals
    MAP3K_t, MAP2K_t, MAPK_t = total_protein
    beta_3, alpha, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # Y_I = Y_t-Y
    if signal_fxn:
        sig = signal_fxn(t,sig,signal_params)

    dMAP3K = (sig/(1+X/beta_3)) * (((k1)*MAP3K_I)/(K_1+MAP3K_I)) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K+alpha*MAPK)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5*MAP2K)*MAPK_I)/(K_5+MAPK_I)) - (k6*MAPK/(K_6+MAPK))
    dX = s7*MAPK - d8*X

    return dMAP3K, dMAP2K, dMAPK, dX

def simulate_wt_experiment(inits, total_protein, sig, learned_params, time, signal_fxn, signal_params):
    # parameters to be learned - inits
    # parameters to be kept constant - params_constants
    # parameters to be learned - learned_params

    #solve odes:
    odes = odeint(b3_a2_1D_X, inits, time, args=(total_protein, sig, learned_params, signal_fxn, signal_params))

    return odes

def simulate_t100a_experiment(inits, total_protein, sig, learned_params, time, signal_fxn, signal_params):
    # parameters to be learned - inits
    # parameters to be kept constant - params_constants
    # parameters to be learned - learned_params
    beta_3, alpha, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6  = learned_params
    learned_params = beta_3, 0, k1, k3, k5, 0, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
    #solve odes:
    odes = odeint(b3_a2_1D_X, inits, time, args=(total_protein, sig, learned_params, signal_fxn, signal_params))

    return odes

def signal_periodic(t_step, signal, period):
    if t_step == 0:
        s = 0
    elif t_step >= 60:
        s = 0
    elif t_step >= 30:
        s = 0
    elif np.floor(t_step / period) % 2 == 0:
        s =  signal
    else:
        s = 0
    return s

experiment_fxns = [simulate_wt_experiment, simulate_t100a_experiment]

#############################################################################
#EA FUNCTIONS
#############################################################################

def convert_individual(ea_individual, conversion_matrix, number_of_params):
    # copy and get len of individual
    arr_params_conv = np.zeros(number_of_params)#np.copy(arr_parameters)
    len_ind = len(ea_individual)

    # Interp:
    for idx in np.nonzero(conversion_matrix[0])[0]:
        ea_val = ea_individual[idx]
        r_min = conversion_matrix[1][idx]
        r_max = conversion_matrix[2][idx]
        arr_params_conv[idx] = np.interp(ea_val, (0,1), (r_min, r_max))

    # Exponentiate:
    for idx in np.nonzero(conversion_matrix[3])[0]:
        ea_val = arr_params_conv[idx]
        base_val = conversion_matrix[4][idx]
        arr_params_conv[idx] = np.power(base_val, ea_val)

    # arr_params_conv[-4:] = np.round(arr_params_conv[-4:],0)

    return arr_params_conv


def scorefxn1(experiment_data, inits, total_protein,
              learned_params, experiment_time):
    mse_total = 0
    arr_params_IP = convert_individual(learned_params, arr_conversion_matrix, number_of_params)

    for dose in experiment_doses:
        exp_data = experiment_data.get(dose)
        exp_time = experiment_time.get(dose)
        # for i, fxn in enumerate(experiment_fxns):
        data = simulate_wt_experiment(inits, total_protein, dose, arr_params_IP, exp_time[0], None, None)
        mapk = data[:,2]/total_protein[2]*100
        error_active = ((exp_data[0] - mapk)**2).mean()
        mse_total += error_active
            # if dose == 550000:
                # map2k = data[:,1]/total_protein[1]*100
                # error_active = ((exp_data[i+2] - map2k)**2).mean()
                # mse_total += error_active
        if dose == 550000:
            data = simulate_wt_experiment(inits, total_protein, dose, arr_params_IP, exp_time[1], signal_periodic, signal_params)
            mapk = data[:,2]/total_protein[2]*100
            error_active = ((exp_data[1] - mapk)**2).mean()
            mse_total += error_active

            # map2k = data[:,1]/total_protein[1]*100
            # error_active = ((exp_data[5] - map2k)**2).mean()
            # mse_total += error_active

    return mse_total


def scorefxn_helper(individual):
    # just a helper function that pulls all of scorefxn1 dependencies together
    # note the (), <--using single optimization in DEAP for now
    # scorefxn1 is taking care of the multiple optimizations for now
    return scorefxn1(experiment_data, inits, total_protein, individual, experiment_time),



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

###################################################################
#LOOP: EVOLUTIONARY ALGORITHM + SAVE DATA
###################################################################
def run():
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
        ind_np_conv = convert_individual(ind_np, arr_conversion_matrix, number_of_params)
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
def throw_away_function(_):
    return run()

def main():
    pool=multiprocessing.Pool(processes=number_of_runs)
    pool.map(throw_away_function, range(number_of_runs))

if __name__ == '__main__':
    main()
