import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
# import seaborn as sns
import pandas as pd
import pathlib
import collections
import h5py
import os
import random

import sys
sys.path.insert(1, '../')
import model_selection as M
# import model_supp

def get_data():
    base = 'C:/Users/sksuzuki/Desktop/killdevil/data/Hybrid_data'

    # base = '/pine/scr/j/m/jmcgirr/sksuzuki/Hybrid_data'
    wt_folder = base + '/WT'
    t100a_folder = base + '/T100A'
    pbs2_folder = base + '/Pbs2'
    pbs2_t100a_folder = base + '/Pbs2_T100A'
    # hog1_ramp_folder = base + '/ramp_1'
    ptpD_folder = base + '/ptpD'

    # wt_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK_activation/WT'
    # t100a_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK_activation/T100A'
    # pbs2_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK_activation/Pbs2'
    # pbs2_t100a_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK_activation/Pbs2_T100A'
    # # ramp_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/ramp_1'
    # ptpD_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK_activation/ptpD'

    mapk_time, mapk_wt_data = load_csv_data(wt_folder)
    mapk_time, mapk_t100a_data = load_csv_data(t100a_folder)
    mapk_time_t100a_long = [0, 30, 60, 90, 120, 150, 180, 240, 300]

    mapk_time, map2k_wt_data = load_csv_data(pbs2_folder)
    mapk_time, map2k_t100a_data = load_csv_data(pbs2_t100a_folder)
    mapk_time, mapk_ptpD_data = load_csv_data(ptpD_folder)
    data = [mapk_wt_data, mapk_t100a_data, map2k_wt_data, map2k_t100a_data, [], mapk_ptpD_data]
    time = [mapk_time, mapk_time_t100a_long, []]
    return data, time

def load_csv_data(folder):
    data = []
    doses = []
    for csv in pathlib.Path(folder).glob('*.csv'):
        f_data = pd.read_csv(csv)
        time = f_data['Time'].tolist()
        dose = f_data['Dose'][0]
        doses.append(dose)
        f_data=f_data.set_index('Time')
        f_data = f_data.iloc[:,:3].mean(axis=1)
        f_data = f_data.tolist()
        data.append(f_data)
    data = np.array(data)
    re_idx = sorted(range(len(doses)), key=lambda k: doses[k])
    data = data[re_idx]
    return time, list(data)

def molarity_conversion(molecules):
    Na = 6.02214076*10**23
    cell_volume = 44                                 # volume of a yeast cell
    return molecules/(Na*cell_volume*10**-15)*1000000 # returns uM

def check_dir_exist():
    stripped_name = strip_filename(save_filename)
    dir_to_use = os.getcwd() + '/' + stripped_name
    #check if dir exists and make
    if not os.path.isdir(dir_to_use):
        os.makedirs(dir_to_use)
        fn = dir_to_use + '/' + 'output.txt'
        file = open(fn, 'w')
        script_name = os.path.basename(__file__)#__file__)
        open_script = open(script_name, 'r')
        write_script = open_script.read()
        file.write(write_script)
        open_script.close()

        file.close()
    return dir_to_use, stripped_name

def get_filename(dir_to_use, stripped_name, val):
    filename_base = dir_to_use + '/' + stripped_name + '_'
    if val < 10:
        toret = '000' + str(val)
    elif 10 <= val < 100:
        toret = '00' + str(val)
    elif 100 <= val < 1000:
        toret = '0' + str(val)
    else:
        toret = str(val)
    return filename_base + toret + '.hdf5'

def strip_filename(fn):
    if '/' in fn:
        fn = fn.split('/')[-1]
    fn = fn.split('.')[0]
    return fn

def add_info(fn, gens, inds, mutationrate, crossoverrate):
    #get current date:
    cur_date = timeski.strftime('%y%m%d')
    # setup EA data:
    ea_data = str(gens) + 'g' + str(inds) + 'i' + str(int(mutationrate*100)) + 'm' + str(int(crossoverrate*100)) + 'c'
    #put it all together:
    #new_fn = cur_date + '_' + fn + '_' + ea_data
    new_fn = cur_date + '_' + os.path.basename(fn).split('.')[0].split('_')[-1] + '_' + ea_data
    return new_fn

def data_to_hdf5(dir_to_use, stripped_name, params, mses, c1=None, c2=None):
    # arr_to_hdf5 = [arr_best_score, arr_best_ind]
    counter = 0
    print(c1, c2)
    filename = get_filename(dir_to_use, stripped_name, counter)
    while os.path.isfile(filename) == True:
        counter += 1
        filename = get_filename(dir_to_use, stripped_name, counter)
    print(filename)
    with h5py.File(filename, 'w') as f:
        f.create_dataset("M2b_mses", data = mses[0])
        f.create_dataset("M2b_params", data = params[0])
        f.create_dataset("M2c_mses", data = mses[1])
        f.create_dataset("M2c_params", data = params[1])
        if not c1 is None:
            f.create_dataset("count1", data=c1)
        if not c2 is None:
            f.create_dataset("count2", data=c2)


def get_schedule(ei):
    N, tau = ei, .25
    # Maximum time to consider (s)
    tmax = 5
    # A suitable grid of time points, and the exponential decay itself
    t = np.linspace(0, tmax, 40)
    y = N * np.exp(-t/tau)
    return y

def select_model(models):
    return random.choice(models)
    # model = models[idx]

def convert_dict_to_array(d):
    return np.array(list(d.values()))

def get_empty_structs():
    M2b_schedule_mses = []
    M2b_schedule_params = []

    M2c_schedule_mses = []
    M2c_schedule_params = []

    c_M2b = collections.Counter({'Pass': 0, 'Fail': 0})
    c_M2c = collections.Counter({'Pass': 0, 'Fail': 0})
    return [M2b_schedule_params, M2c_schedule_params], [M2b_schedule_mses, M2c_schedule_mses], c_M2b, c_M2c

def eval_run(mse, params, M2b_params, M2c_params, M2b_mses, M2c_mses, c_M2b, c_M2c, idx, ei):
    if idx == 0:
        if mse < ei:
            c_M2b['Pass'] += 1
            M2b_mses.append(mse)
            M2b_params.append(params)
            print(c_M2b)
        else:
            c_M2b['Fail'] += 1
    if idx == 1:
        if mse < ei:
            c_M2c['Pass'] += 1
            M2c_mses.append(mse)
            M2c_params.append(params)
            print(c_M2c)
        else:
            c_M2c['Fail'] += 1
    return M2b_params, M2c_params, M2b_mses, M2c_mses, c_M2b, c_M2c

def run_first_schedule(model_fxns, ei, particle_num):
    params, mses, c_M2b, c_M2c = get_empty_structs()
    M2b_params, M2c_params = params
    M2b_mses, M2c_mses = mses
    while len(mses[0]+mses[1]) < particle_num:
        model = select_model(model_fxns)
        idx = model_fxns.index(model)
        params = model.select_params()
        mse = sum(model.calc_sim_score(params, exp_data, exp_time, ptpD=False)[:18])
        M2b_params, M2c_params, M2b_mses, M2c_mses, c_M2b, c_M2c = eval_run(mse, params, M2b_params, M2c_params, M2b_mses, M2c_mses, c_M2b, c_M2c, idx, ei)

    params = [M2b_params, M2c_params]
    mses = [M2b_mses, M2c_mses]
    c_M2b = convert_dict_to_array(c_M2b)
    c_M2c = convert_dict_to_array(c_M2c)

    return params, mses, c_M2b, c_M2c


def run_priored_schedule(model_fxns,prior_params, ei, particle_num):
    params, mses, c_M2b, c_M2c = get_empty_structs()
    M2b_params, M2c_params = params
    M2b_mses, M2c_mses = mses
    while len(mses[0]+mses[1]) < particle_num:
        model = select_model(model_fxns)
        idx = model_fxns.index(model)
        model.random_select = False
        priors = prior_params[idx]
        params = model.select_params(priors)
        new_params = model.step_params(params)
        mse = sum(model.calc_sim_score(new_params, exp_data, exp_time, ptpD=False)[:18])
        M2b_params, M2c_params, M2b_mses, M2c_mses, c_M2b, c_M2c = eval_run(mse, new_params, M2b_params, M2c_params, M2b_mses, M2c_mses, c_M2b, c_M2c, idx, ei)

    params = [M2b_params, M2c_params]
    mses = [M2b_mses, M2c_mses]
    c_M2b = convert_dict_to_array(c_M2b)
    c_M2c = convert_dict_to_array(c_M2c)

    return params, mses, c_M2b, c_M2c


def main(particle_num, model_fxns, eis):
    dir_to_use, stripped_name = check_dir_exist()


    params, mses, c_M2b, c_M2c = run_first_schedule(model_fxns, eis[0], particle_num)
    data_to_hdf5(dir_to_use, stripped_name, params, mses, c_M2b, c_M2c)
    for i, ei in enumerate(eis[1:]):
        print("Now running schedule #" + str(i+1))
        params, mses, c_M2b, c_M2c = run_priored_schedule(model_fxns, params, ei, particle_num)
        data_to_hdf5(dir_to_use, stripped_name, params, mses, c_M2b, c_M2c)

if __name__ == '__main__':
    exp_data, exp_time = get_data()

    MAP3K_t = molarity_conversion(701)
    MAP2K_t = molarity_conversion(2282)
    MAPK_t = molarity_conversion(5984)
    PTP_t = molarity_conversion(118+400) # including ptc1

    MAP3K = 0.05*MAP3K_t # estimated (so not 0)
    MAP2K = 0.05975380333*MAP2K_t # from the biological data
    MAPK = 0.00540042381*MAPK_t  # from the biological data
    gly = 0.00001 # placeholder (so not 0)
    PTP = molarity_conversion(118+400) # start with all on

    # doses
    hog1_doses = [0, 50000, 150000, 250000, 350000, 450000, 550000]
    # pbs2_doses = [150000, 550000]
    # ptp_doses = [0, 150000, 550000]
    inits = [MAP3K, MAP2K, MAPK, gly]
    total_protein = [MAP3K_t, MAP2K_t, MAPK_t, 550000*2] #uM, except for gly (1) which is a placeholder for multiplying arrays together

    #
    # minimums = [-4, -4, -4,
    #     -4, -4, -4, -4,
    #     -4, -4, -4, -4, -4,
    #     -4, -4, -4, -4,
    #     -4]
    #
    # maximums = [ 4, 4, 4,
    #     4, 4, 4, 4,
    #     4, 4, 4, 4, 4,
    #     4, 4, 4, 4,
    #     4]
    min_M2b = np.array([-4.        , -3.99979862, -4.        , -4.        , -0.15473962,
        0.06426045, -3.98909537, -2.83016735, -1.96762929, -1.8346866 ,
       -4.        , -4.        , -4.        , -4.        , -4.        ,
       -4.        , -3.99999997])

    max_M2b = np.array([ 3.99533189,  3.05750845,  4.        ,  4.        ,  4.        ,
        4.        ,  4.        ,  4.        ,  4.        ,  4.        ,
       -1.50184403,  4.        ,  3.01964589,  2.90589447,  4.        ,
        4.        ,  4.        ])

    min_M2c = np.array([-4.        , -3.99999853, -4.        , -4.        , -0.44020075,
        0.05658478, -4.        , -2.79419613, -1.98155364, -1.39947218,
       -4.        , -4.        , -4.        , -4.        , -4.        ,
       -4.        , -4.        ])
    max_M2c = np.array([ 4.        ,  3.56337657,  4.        ,  4.        ,  4.        ,
        4.        ,  4.        ,  4.        ,  4.        ,  4.        ,
       -1.47121131,  4.        ,  3.51918546,  3.20807053,  4.        ,
        4.        ,  3.99999991])

    exp_base = 'C:/Users/sksuzuki/Desktop/killdevil/data/Hybrid_data'

    ei = 50000#(9*9*100)**2
    eis = get_schedule(ei)
# define model functions
    # model_fxn = model.Model(model.M2_kb, model.simulate_t100a_experiment_M2_kb, model.simulate_nopos_experiment_M2a_kb)
    # model_fxn = model.Model(model.M2_kb, model.simulate_t100a_experiment_M2_kb, model.simulate_nopos_experiment_M2a_kb)
    model_fxn1 = M.Model(M.M2b, M.simulate_t100a_experiment_M2_, min_M2b, max_M2b, inits, total_protein, hog1_doses)
    model_fxn2 = M.Model(M.M2c, M.simulate_t100a_experiment_M2_, min_M2c, max_M2c, inits, total_protein, hog1_doses)

    model_fxns = [model_fxn1, model_fxn2]
    save_filename = 't200328_M2b_M2c_model_selection.txt'
    # number_eas = 500
    particle_num = 5000
    # main('/pine/scr/s/k/sksuzuki/HOG_model/paper/190924_kb_M2/', number_eas, particle_num)
    main(particle_num, model_fxns, eis)
