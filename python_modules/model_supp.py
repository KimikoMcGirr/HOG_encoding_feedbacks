import pathlib
import pickle
import numpy as np
import pickle
import os
import pandas as pd
import model
import h5py

def molarity_conversion(molecules):
    Na = 6.02214076*10**23
    cell_volume = 44                                 # volume of a yeast cell
    return molecules/(Na*cell_volume*10**-15)*1000000 # returns uM

def get_sim_data(folder, num_sims=None):
    try:
        my_abs_path = os.path.exists(folder)
    except FileNotFoundError:
        print("Folder not found ¯\_(ツ)_/¯")
    else:
        all_mses = []
        all_params = []
        empty_data = 0
        for i, loaded_data in enumerate(pathlib.Path(folder).glob('*.pickled')):
            if num_sims:
                if i > num_sims-1:
                    break
            if os.path.getsize(loaded_data) > 0:
            # with open(str(loaded_data), 'rb') as f: # python35
                with open(loaded_data, 'rb') as f:
                    new_data = pickle.load(f)
                    all_mses.append(np.asarray(new_data[0]))
                    all_params.append(np.asarray(new_data[1]))
            else:
                empty_data += 1

    print("Number of runs collected: " + str(len(all_params)))

    last_mses = [all_mses[i][len(all_mses[-1])-1] for i in range(len(all_mses))]
    last_params = [all_params[i][len(all_params[-1])-1] for i in range(len(all_params))]

    last_mses = np.array(last_mses)
    last_params = np.array(last_params)
    all_mses = np.array(all_mses)
    all_params = np.array(all_params)

    print('Best last gen MSE: ' + str(np.sort(last_mses)[0]))
    print('Mean last gen MSEs of top 5%: ' + str(np.mean(np.sort(last_mses)[:round(len(last_mses)*0.05)])))
    return all_params, last_params, all_mses, last_mses

def load_Justin_csv_data(folder):
    data = []
    doses = []
    for csv in pathlib.Path(folder).glob('*.csv'):
        f_data = pd.read_csv(csv)
        time = f_data['Time'].tolist()
        if 'Dose' in f_data:
            dose = f_data['Dose'][0]
            doses.append(dose)
        f_data=f_data.set_index('Time')
        f_data = f_data.iloc[:,:3].mean(axis=1)
        f_data = f_data.tolist()
        data.append(f_data)
    data = np.array(data)
    if 'Dose' in f_data:
        re_idx = sorted(range(len(doses)), key=lambda k: doses[k])
        data = data[re_idx]
    return time, list(data)

# def get_data(): #AJDUST
#     wt_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/WT'
#     t100a_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/T100A'
#     pbs2_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/Pbs2'
#     pbs2_t100a_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/Pbs2_T100A'
#     # sho1DD_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/sho1DD'
#     # ssk1D_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/ssk1D'
#     hog1_ramp_folder =  'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/ramp_1'
#     # pbs2_ramp_folder =  'C:/Users/sksuzuki/Desktop/killdevil/data_pbs2/MAPK activation/pulse_pbs2'
#     ptpD_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/ptpD'
#
#     mapk_time, mapk_wt_data = load_csv_data(wt_folder)
#     mapk_time, mapk_t100a_data = load_csv_data(t100a_folder)
#     # mapk_data_t100a_long = [mapk_t100a_data[0]]
#     mapk_time_t100a_long = [0, 2, 5, 10, 15, 20, 25, 30, 60, 90, 120, 150, 180, 240, 300]
#
#     mapk_time, map2k_wt_data = load_csv_data(pbs2_folder)
#     mapk_time, map2k_t100a_data = load_csv_data(pbs2_t100a_folder)
#     mapk_ramp_time, hog1_ramp_data = load_csv_data(hog1_ramp_folder)
#     mapk_time, mapk_ptpD_data = load_csv_data(ptpD_folder)
#     # mapk_time, sho1_wt_data = load_csv_data(ssk1D_folder)
#     # mapk_time, sln1_wt_data = load_csv_data(sho1DD_folder)
#     data = [mapk_wt_data, mapk_t100a_data, map2k_wt_data, map2k_t100a_data, hog1_ramp_data, mapk_ptpD_data]
#     time = [mapk_time, mapk_time_t100a_long, mapk_ramp_time]
#     return data, time
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
    # print(doses)
    re_idx = sorted(range(len(doses)), key=lambda k: doses[k])
    data = data[re_idx]
    return time, list(data)

def get_data(local=False): #AJDUST
    if local:
        base = '../../exp_data/phospho_data'
    else:
        base = '/nas/longleaf/home/sksuzuki/data/paper'
    wt_folder = base + '/WT'
    t100a_folder = base + '/T100A'
    pbs2_folder = base + '/Pbs2'
    pbs2_t100a_folder = base + '/Pbs2_T100A'
    # sho1DD_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/sho1DD'
    # ssk1D_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/ssk1D'
    hog1_ramp_folder =  base + '/ramp_1'
    hog1_ramp_inhib_folder =  base + '/ramp_1_inhib'
    pbs2_ramp_folder = base + '/ramp_1_pbs2'
    ptpD_folder = base + '/ptpD'

    mapk_time, mapk_wt_data = load_csv_data(wt_folder)
    mapk_time, mapk_t100a_data = load_csv_data(t100a_folder)
    # mapk_data_t100a_long = [mapk_t100a_data[0]]
    mapk_time_t100a_long = [0, 30, 60, 90, 120, 150, 180, 240, 300]

    mapk_time, map2k_wt_data = load_csv_data(pbs2_folder)
    mapk_time, map2k_t100a_data = load_csv_data(pbs2_t100a_folder)
    mapk_ramp_time, hog1_ramp_data = load_csv_data(hog1_ramp_folder)
    mapk_ramp_time, hog1_ramp_inhib_data = load_csv_data(hog1_ramp_inhib_folder)
    mapk_ramp_time, pbs2_ramp_data = load_csv_data(pbs2_ramp_folder)
    mapk_time, mapk_ptpD_data = load_csv_data(ptpD_folder)
    # mapk_time, sho1_wt_data = load_csv_data(ssk1D_folder)
    # mapk_time, sln1_wt_data = load_csv_data(sho1DD_folder)
    # data = [mapk_wt_data, mapk_t100a_data, map2k_wt_data, map2k_t100a_data] #,
    data = [mapk_wt_data, mapk_t100a_data, map2k_wt_data, map2k_t100a_data, hog1_ramp_data, hog1_ramp_inhib_data, pbs2_ramp_data, mapk_ptpD_data]
    time = [mapk_time, mapk_time_t100a_long, mapk_ramp_time]
    return data, time

def get_Justin_data(): #AJDUST
    wt_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/WT'
    t100a_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/T100A'

    mapk_time, mapk_wt_data = load_Justin_csv_data(wt_folder)
    mapk_time, mapk_t100a_data = load_Justin_csv_data(t100a_folder)

    data = [mapk_wt_data, mapk_t100a_data]
    time = [mapk_time]
    return data, time

def get_my_350_data(): #AJDUST
    # wt_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK activation/WT'
    t100a_folder = 'C:/Users/sksuzuki/Desktop/killdevil/data/MAPK_activation_350/T100A'

    # mapk_time, mapk_wt_data = load_Justin_csv_data(wt_folder)
    mapk_time, mapk_t100a_data = load_Justin_csv_data(t100a_folder)

    data = mapk_t100a_data
    time = mapk_time
    return data, time

def sim_all(model_fxns, exp_data, exp_time, total_protein, inits, params, params_step, ptpD=False, full=False, convert=''):
    if convert[0]:
        params = convert_individual(params, convert[1], len(params))
    # mapk_wt_data, mapk_t100a_data, map2k_wt_data, map2k_t100a_data, hog1_ramp_data, mapk_ptpD_data = data
    # mapk_data_t100a_long = [mapk_t100a_data[0]]
    # mapk_time, mapk_time_t100a_long, mapk_ramp_time = exp_time
    # hog1_doses = [0, 50000, 150000, 250000, 350000, 450000, 550000]
    wt_doses = [0, 50000, 150000,250000, 350000, 450000, 550000]
    # t100a_doses = [0, 150000,550000] same as wt


    # exp_data, exp_time = get_data()
    #
    mapk_wt_data, mapk_t100a_data, map2k_wt_data, map2k_t100a_data, hog1_ramp_data, hog1_ramp_inhib_data, pbs2_ramp_data, mapk_ptpD_data = exp_data
    mapk_time, mapk_time_t100a_long, mapk_ramp_time = exp_time

    # mapk_data_t100a_0 = [mapk_t100a_data[0]]
    # mapk_time_t100a_0 = [0, 30, 60, 90, 120, 150, 180, 240, 300]

    wt_ss_inits = model.run_ss(model_fxns.m, inits, total_protein, params)

    dt = 0.1
    steps = 601
    time = np.linspace(0,dt*steps,steps)
    time_long = np.linspace(0,dt*3001,steps)

    # closest_idxs_mapk = [np.abs(time - t).argmin() for t in mapk_time]
    # closest_idxs_t100a_long = [np.abs(time_long - t).argmin() for t in mapk_time_t100a_0]
    # closest_idxs_ramp_time = [np.abs(time - t).argmin() for t in mapk_ramp_time]


#     mse_total = 0
#         # ptpDs
#     if ptpD:
#         mses = np.zeros(23)
#         ptp_doses = [0, 150000, 350000, 550000]
#         ptpD_total_protein = total_protein[:-1] + [0]
#         ptpD_inits = inits[:-1] + [0]
#
#         ptpD_ss_inits = model.run_ss(model_fxns.m, ptpD_inits, ptpD_total_protein, params)
#         # ptpD_ss_inits = model.run_ss_ptps(model_fxns.m, inits, total_protein, params)
#
#         for i, (dose, data) in enumerate(zip(ptp_doses, mapk_ptpD_data), 19):
#             odes = model.simulate_wt_experiment(model_fxns.m, ptpD_ss_inits, ptpD_total_protein, dose, params, time)
#             odes = model.simulate_wt_experiment(model_fxns.m, ptpD_ss_inits, ptpD_total_protein, dose, params, time)
#             # odes = model.simulate_ptpD_experiment(model_fxns.m, ptpD_ss_inits, total_protein, dose, params, time)
#             mapk = odes[:,2]/total_protein[2]*100
#             mses[i] = np.sum((data - mapk[closest_idxs_mapk])**2)
#             mse_total += mses[i]
#
#     # WILDTYPE
#     # Hog1
#     else:
#         mses = np.zeros(19)
#
#     for i, (dose, data) in enumerate(zip(wt_doses, mapk_wt_data), 0):
#         odes = model.simulate_wt_experiment(model_fxns.m, wt_ss_inits, total_protein, dose, params, time)#mapk_time)
#         mapk = odes[:,2]/total_protein[2]*100
#         mses[i] = np.sum((data - mapk[closest_idxs_mapk])**2)
#         mse_total += mses[i]
#
#         # Pbs2
#         if dose == 150000:
#             map2k = odes[:,1]/total_protein[1]*100
#             mses[14] = np.sum((map2k_wt_data[0] - map2k[closest_idxs_mapk])**2)
#             mse_total += mses[14]
#         elif dose == 550000:
#             map2k = odes[:,1]/total_protein[1]*100
#             mses[15] = np.sum((map2k_wt_data[1] - map2k[closest_idxs_mapk])**2)
#             mse_total += mses[15]
# #     (mse_total/63)
#
#     # ANALOG SENSITIVE
#     # Hog1
#     for i, (dose, data) in enumerate(zip(wt_doses, mapk_t100a_data), 7):
#         if dose == 0:
#             odes = model_fxns.t100a(model_fxns.m, wt_ss_inits, total_protein, dose, params, time_long)
#             mapk = odes[:,2]/total_protein[2]*100
#             mses[i] = np.sum((data - mapk[closest_idxs_t100a_long])**2)
#             mse_total += mses[i]
#         else:
#             odes = model_fxns.t100a(model_fxns.m, wt_ss_inits, total_protein, dose, params, time)
#             mapk = odes[:,2]/total_protein[2]*100
#             mses[i] = np.sum((data - mapk[closest_idxs_mapk])**2)
#             mse_total += mses[i]
#             # Pbs2
#             if dose == 150000:
#                 map2k = odes[:,1]/total_protein[1]*100
#                 mses[16] = np.sum((map2k_t100a_data[0] - map2k[closest_idxs_mapk])**2)
#                 mse_total += mses[16]
#             elif dose == 550000:
#                 map2k = odes[:,1]/total_protein[1]*100
#                 mses[17] = np.sum((map2k_t100a_data[1] - map2k[closest_idxs_mapk])**2)
#                 mse_total += mses[17]

    # Hog1 ramp
    # for state in states:

    odes = model.simulate_wt_experiment(model_fxns.m, wt_ss_inits, total_protein, 0, params, time, run_type=['ramp'])
    # mapk = odes[:,2]/total_protein[2]*100

    odes_step = model.simulate_wt_experiment(model_fxns.m, wt_ss_inits, total_protein, 0, params_step, time, run_type=['ramp'])
    # mapk_step = odes[:,2]/total_protein[2]*100
        # mses[18] = ((data - mapk[closest_idxs_ramp_time])**2).mean()
        # mses[18] = sum((data - mapk[closest_idxs_ramp_time])**2)
    diff_odes = np.abs(np.sum(odes-odes_step, axis=1))
    print(diff_odes)
#     (mse_total/13)
    #     (mse_total/27)
    # print(mses)
    # print("Hog1 fit to WT: " + str(sum(mses[:7])))
    # print("Hog1 fit to T100A: " + str(sum(mses[7:14])))
    # print("Pbs2 fit to WT: " + str(sum(mses[14:16])))
    # print("Pbs2 fit to T100A: " + str(sum(mses[16:18])))
    # print("Hog1 fit to Ramp: " + str(mses[18]))
    # if ptpD:
    #     print("Hog1 fit to PtpD: " + str(sum(mses[19:])))

    # return mses
    # if ptpD:
    #     return sum(mses[:18])+mses[19]
    # elif full:
    #     # print(mses)
    #     return mses
    # else:
    #     return sum(mses[:18])


def calc_sim_score(model_fxns, exp_data, exp_time, total_protein, inits, params, ptpD=False, full=False, convert=''):
    if convert[0]:
        params = convert_individual(params, convert[1], len(params))
    # mapk_wt_data, mapk_t100a_data, map2k_wt_data, map2k_t100a_data, hog1_ramp_data, mapk_ptpD_data = data
    # mapk_data_t100a_long = [mapk_t100a_data[0]]
    # mapk_time, mapk_time_t100a_long, mapk_ramp_time = exp_time
    # hog1_doses = [0, 50000, 150000, 250000, 350000, 450000, 550000]
    wt_doses = [0, 50000, 150000,250000, 350000, 450000, 550000]
    # t100a_doses = [0, 150000,550000] same as wt


    # exp_data, exp_time = get_data()
    #
    mapk_wt_data, mapk_t100a_data, map2k_wt_data, map2k_t100a_data, hog1_ramp_data, hog1_ramp_inhib_data, pbs2_ramp_data, mapk_ptpD_data = exp_data
    mapk_time, mapk_time_t100a_long, mapk_ramp_time = exp_time

    # mapk_data_t100a_0 = [mapk_t100a_data[0]]
    mapk_time_t100a_0 = [0, 30, 60, 90, 120, 150, 180, 240, 300]

    wt_ss_inits = model.run_ss(model_fxns.m, inits, total_protein, params)

    dt = 0.1
    steps = 601
    time = np.linspace(0,dt*steps,steps)
    time_long = np.linspace(0,dt*3001,steps)

    closest_idxs_mapk = [np.abs(time - t).argmin() for t in mapk_time]
    closest_idxs_t100a_long = [np.abs(time_long - t).argmin() for t in mapk_time_t100a_0]
    closest_idxs_ramp_time = [np.abs(time - t).argmin() for t in mapk_ramp_time]


    mse_total = 0
        # ptpDs
    if ptpD:
        mses = np.zeros(23)
        ptp_doses = [0, 150000, 350000, 550000]
        ptpD_total_protein = total_protein[:-1] + [0]
        ptpD_inits = inits[:-1] + [0]

        ptpD_ss_inits = model.run_ss(model_fxns.m, ptpD_inits, ptpD_total_protein, params)
        # ptpD_ss_inits = model.run_ss_ptps(model_fxns.m, inits, total_protein, params)

        for i, (dose, data) in enumerate(zip(ptp_doses, mapk_ptpD_data), 19):
            odes = model.simulate_wt_experiment(model_fxns.m, ptpD_ss_inits, ptpD_total_protein, dose, params, time)
            # odes = model.simulate_ptpD_experiment(model_fxns.m, ptpD_ss_inits, total_protein, dose, params, time)
            mapk = odes[:,2]/total_protein[2]*100
            mses[i] = np.sum((data - mapk[closest_idxs_mapk])**2)
            mse_total += mses[i]

    # WILDTYPE
    # Hog1
    else:
        mses = np.zeros(19)

    for i, (dose, data) in enumerate(zip(wt_doses, mapk_wt_data), 0):
        odes = model.simulate_wt_experiment(model_fxns.m, wt_ss_inits, total_protein, dose, params, time)#mapk_time)
        mapk = odes[:,2]/total_protein[2]*100
        mses[i] = np.sum((data - mapk[closest_idxs_mapk])**2)
        mse_total += mses[i]

        # Pbs2
        if dose == 150000:
            map2k = odes[:,1]/total_protein[1]*100
            mses[14] = np.sum((map2k_wt_data[0] - map2k[closest_idxs_mapk])**2)
            mse_total += mses[14]
        elif dose == 550000:
            map2k = odes[:,1]/total_protein[1]*100
            mses[15] = np.sum((map2k_wt_data[1] - map2k[closest_idxs_mapk])**2)
            mse_total += mses[15]
#     (mse_total/63)

    # ANALOG SENSITIVE
    # Hog1
    for i, (dose, data) in enumerate(zip(wt_doses, mapk_t100a_data), 7):
        if dose == 0:
            odes = model_fxns.t100a(model_fxns.m, wt_ss_inits, total_protein, dose, params, time_long)
            mapk = odes[:,2]/total_protein[2]*100
            mses[i] = np.sum((data - mapk[closest_idxs_t100a_long])**2)
            mse_total += mses[i]
        else:
            odes = model_fxns.t100a(model_fxns.m, wt_ss_inits, total_protein, dose, params, time)
            mapk = odes[:,2]/total_protein[2]*100
            mses[i] = np.sum((data - mapk[closest_idxs_mapk])**2)
            mse_total += mses[i]
            # Pbs2
            if dose == 150000:
                map2k = odes[:,1]/total_protein[1]*100
                mses[16] = np.sum((map2k_t100a_data[0] - map2k[closest_idxs_mapk])**2)
                mse_total += mses[16]
            elif dose == 550000:
                map2k = odes[:,1]/total_protein[1]*100
                mses[17] = np.sum((map2k_t100a_data[1] - map2k[closest_idxs_mapk])**2)
                mse_total += mses[17]

    # Hog1 ramp
    for data in hog1_ramp_data:
        odes = model.simulate_wt_experiment(model_fxns.m, wt_ss_inits, total_protein, 0, params, time, run_type=['ramp'])
        mapk = odes[:,2]/total_protein[2]*100
        # mses[18] = ((data - mapk[closest_idxs_ramp_time])**2).mean()
        mses[18] = sum((data - mapk[closest_idxs_ramp_time])**2)

#     (mse_total/13)
    #     (mse_total/27)
    # print(mses)
    # print("Hog1 fit to WT: " + str(sum(mses[:7])))
    # print("Hog1 fit to T100A: " + str(sum(mses[7:14])))
    # print("Pbs2 fit to WT: " + str(sum(mses[14:16])))
    # print("Pbs2 fit to T100A: " + str(sum(mses[16:18])))
    # print("Hog1 fit to Ramp: " + str(mses[18]))
    # if ptpD:
    #     print("Hog1 fit to PtpD: " + str(sum(mses[19:])))

    # return mses
    if ptpD:
        return sum(mses[:18])+mses[19]
    elif full:
        # print(mses)
        return mses
    else:
        return sum(mses[:18])

def calc_sim_score_M2c_ptp(model_fxns, exp_data, exp_time, total_protein, inits, params, ptpD=False, full=False, convert=''):
    if convert[0]:
        params = convert_individual(params, convert[1], len(params))
    # mapk_wt_data, mapk_t100a_data, map2k_wt_data, map2k_t100a_data, hog1_ramp_data, mapk_ptpD_data = data
    # mapk_data_t100a_long = [mapk_t100a_data[0]]
    # mapk_time, mapk_time_t100a_long, mapk_ramp_time = exp_time
    # hog1_doses = [0, 50000, 150000, 250000, 350000, 450000, 550000]
    wt_doses = [0, 50000, 150000,250000, 350000, 450000, 550000]
    # t100a_doses = [0, 150000,550000] same as wt


    # exp_data, exp_time = get_data()
    #
    mapk_wt_data, mapk_t100a_data, map2k_wt_data, map2k_t100a_data, hog1_ramp_data, hog1_ramp_inhib_data, pbs2_ramp_data, mapk_ptpD_data = exp_data
    mapk_time, mapk_time_t100a_long, mapk_ramp_time = exp_time

    # mapk_data_t100a_0 = [mapk_t100a_data[0]]
    mapk_time_t100a_0 = [0, 30, 60, 90, 120, 150, 180, 240, 300]

    wt_ss_inits = model.run_ss(model_fxns.m, inits, total_protein, params)

    dt = 0.1
    steps = 601
    time = np.linspace(0,dt*steps,steps)
    time_long = np.linspace(0,dt*3001,steps)

    closest_idxs_mapk = [np.abs(time - t).argmin() for t in mapk_time]
    closest_idxs_t100a_long = [np.abs(time_long - t).argmin() for t in mapk_time_t100a_0]
    closest_idxs_ramp_time = [np.abs(time - t).argmin() for t in mapk_ramp_time]


    mse_total = 0
        # ptpDs
    if ptpD:
        mses = np.zeros(23)
        ptp_doses = [0, 150000, 350000, 550000]
        # ptpD_total_protein = total_protein[:-1] + [0]
        # ptpD_inits = inits[:-1] + [0]

        ptpD_ss_inits = model.M2c_run_ptpD_ss(model_fxns.m, ptpD_inits, ptpD_total_protein, params)
        # ptpD_ss_inits = model.run_ss_ptps(model_fxns.m, inits, total_protein, params)

        for i, (dose, data) in enumerate(zip(ptp_doses[0], mapk_ptpD_data[0]), 19):
            odes = model.M2c_simulate_ptpD_experiment(model_fxns.m, ptpD_ss_inits, total_protein, dose, params, time)
            # odes = model.simulate_ptpD_experiment(model_fxns.m, ptpD_ss_inits, total_protein, dose, params, time)
            mapk = odes[:,2]/total_protein[2]*100
            mses[i] = np.sum((data - mapk[closest_idxs_mapk])**2)
            mse_total += mses[i]

    # WILDTYPE
    # Hog1
    else:
        mses = np.zeros(19)

    for i, (dose, data) in enumerate(zip(wt_doses, mapk_wt_data), 0):
        odes = model.M2c_simulate_wt_experiment(model_fxns.m, wt_ss_inits, total_protein, dose, params, time)#mapk_time)
        mapk = odes[:,2]/total_protein[2]*100
        mses[i] = np.sum((data - mapk[closest_idxs_mapk])**2)
        mse_total += mses[i]

        # Pbs2
        if dose == 150000:
            map2k = odes[:,1]/total_protein[1]*100
            mses[14] = np.sum((map2k_wt_data[0] - map2k[closest_idxs_mapk])**2)
            mse_total += mses[14]
        elif dose == 550000:
            map2k = odes[:,1]/total_protein[1]*100
            mses[15] = np.sum((map2k_wt_data[1] - map2k[closest_idxs_mapk])**2)
            mse_total += mses[15]
#     (mse_total/63)

    # ANALOG SENSITIVE
    # Hog1
    for i, (dose, data) in enumerate(zip(wt_doses, mapk_t100a_data), 7):
        if dose == 0:
            odes = model_fxns.t100a(model_fxns.m, wt_ss_inits, total_protein, dose, params, time_long)
            mapk = odes[:,2]/total_protein[2]*100
            mses[i] = np.sum((data - mapk[closest_idxs_t100a_long])**2)
            mse_total += mses[i]
        else:
            odes = model_fxns.t100a(model_fxns.m, wt_ss_inits, total_protein, dose, params, time)
            mapk = odes[:,2]/total_protein[2]*100
            mses[i] = np.sum((data - mapk[closest_idxs_mapk])**2)
            mse_total += mses[i]
            # Pbs2
            if dose == 150000:
                map2k = odes[:,1]/total_protein[1]*100
                mses[16] = np.sum((map2k_t100a_data[0] - map2k[closest_idxs_mapk])**2)
                mse_total += mses[16]
            elif dose == 550000:
                map2k = odes[:,1]/total_protein[1]*100
                mses[17] = np.sum((map2k_t100a_data[1] - map2k[closest_idxs_mapk])**2)
                mse_total += mses[17]

    # Hog1 ramp
    for data in hog1_ramp_data:
        odes = model.M2c_simulate_wt_experiment(model_fxns.m, wt_ss_inits, total_protein, 0, params, time, run_type=['ramp'])
        mapk = odes[:,2]/total_protein[2]*100
        # mses[18] = ((data - mapk[closest_idxs_ramp_time])**2).mean()
        mses[18] = sum((data - mapk[closest_idxs_ramp_time])**2)

#     (mse_total/13)
    #     (mse_total/27)
    # print(mses)
    # print("Hog1 fit to WT: " + str(sum(mses[:7])))
    # print("Hog1 fit to T100A: " + str(sum(mses[7:14])))
    # print("Pbs2 fit to WT: " + str(sum(mses[14:16])))
    # print("Pbs2 fit to T100A: " + str(sum(mses[16:18])))
    # print("Hog1 fit to Ramp: " + str(mses[18]))
    # if ptpD:
    #     print("Hog1 fit to PtpD: " + str(sum(mses[19:])))

    # return mses
    # if ptpD:
    return sum(mses[:18])+mses[19]
    # elif full:
        # print(mses)
        # return mses
    # else:
        # return sum(mses[:18])


def get_mse_stats(model_fxns, param_sets, total_protein, inits, ptpD=True):
    data, time = get_data()
    if ptpD:
        mses = np.zeros([len(param_sets), 22]) #6
    else:
        mses = np.zeros([len(param_sets), 19])
    # print(mses.size)
    for i in range(len(param_sets)):
        mses[i] = calc_sim_score(model_fxns, param_sets[i], data, time, total_protein, inits, ptpD)
    mses = pd.DataFrame(data=mses) #, columns=['WT Hog1', 'Hog1-as Hog1', 'WT Pbs2', 'Hog1-as Pbs2', 'Ramp', 'Ptp2/3$\Delta$']
    return mses

def get_saved_thetas(f):
    # df = pd.read_csv(str(f))
    # df = df.drop(df.columns[0], axis=1)
    return np.array(pd.read_csv(f).drop(['Unnamed: 0'], axis=1))


def sort_mses_thetas(mses, thetas):
    re_idx = sorted(range(len(mses)), key=lambda k: mses[k])
    if type(thetas) == h5py._hl.dataset.Dataset:
        thetas = [thetas[x] for x in re_idx]
    else:
        thetas = thetas[re_idx]
    return np.sort(mses), thetas

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

def make_conversion_matrix(number_of_params, minimums, maximums):
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

    for i in range(len(minimums)):
        arr_IandP[1,i] = minimums[i] #interp_range_min
        arr_IandP[2,i] = maximums[i] #interp_range_max

    return arr_IandP
