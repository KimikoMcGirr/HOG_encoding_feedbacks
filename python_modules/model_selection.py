from scipy.optimize import fsolve
from scipy.integrate import odeint
import random
import numpy as np


class Model:
    def __init__(self, m, t100a,
                minimums, maximums,
                inits, total_protein,
                hog1_doses):

        # model functions
        self.m = m
        self.t100a = t100a

        # model parameters ranges
        self.minimums = minimums
        self.maximums = maximums
        # arr_conversion_matrix = make_conversion_matrix()

        # model parameters
        self.inits = inits
        self.total_protein = total_protein
        # self.learned_params = learned_params

        # doses
        self.hog1_doses = hog1_doses

        self.random_select = False

        self.conversion_matrix = np.array([])

        # self.ptpD_doses = [0]

    def run_ss(self, learned_params):
        ss = fsolve(self.m, self.inits, args=(0,self.total_protein, 0, learned_params))
        return ss

    def simulate_wt_experiment(self, m, inits, total_protein, sig, learned_params, time, run_type=None):
        odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
        return odes

    def signal_ramp_special(t_step):
        sig = 0
        if t_step >= .001:
            sig = 250000
        if t_step >= 20:
            sig = 550000
        return sig

    def make_conversion_matrix(self):
        # want easily savable matrix to hold this info
        # interp boolean, interp range (min,max), power boolean, power number (y)
        arr_IandP = np.zeros((5,len(self.minimums)))
        # Set all interp booleans to 1 - everything is going to be interpreted
        arr_IandP[0,:] = 1
        # Set all power booleans to 1 - everything is in the form of powers
        arr_IandP[3,:] = 1
        # Set all power numbers to 10 - everything has a base of 10
        arr_IandP[4,:] = 10
        # Set minimums and maximums for all parameters. Parameters are in the following order:
        for i in range(len(self.minimums)):
            arr_IandP[1,i] = self.minimums[i] #interp_range_min
            arr_IandP[2,i] = self.maximums[i] #interp_range_max

        return arr_IandP

    def convert_individual(self, pre_param):
        # copy and get len of individual
        if self.conversion_matrix.size == 0:
            self.conversion_matrix = self.make_conversion_matrix()

        post_param = np.zeros(len(pre_param))#np.copy(arr_parameters)
        len_ind = len(pre_param)

        # Interp:
        for idx in np.nonzero(self.conversion_matrix[0])[0]:
            ea_val = pre_param[idx]
            r_min = self.conversion_matrix[1][idx]
            r_max = self.conversion_matrix[2][idx]
            post_param[idx] = np.interp(ea_val, (0,1), (r_min, r_max))

        # Exponentiate:
        for idx in np.nonzero(self.conversion_matrix[3])[0]:
            ea_val = post_param[idx]
            base_val = self.conversion_matrix[4][idx]
            post_param[idx] = np.power(base_val, ea_val)

        return post_param

    # def draw_thetas(self, params):
    #     idx = np.random.choice(range(len(params)), 1)
    #     return sorted_params[idx][0]
    def select_params(self, prior_params=None):
        if self.random_select:
            pre_param = [random.random() for i in range(len(self.minimums))]
            post_param = self.convert_individual(pre_param)
            return post_param
        else:
            return random.choice(prior_params)

    def step_params(self, params):
        log_theta = np.log10(params)
        theta_prime = np.concatenate([10**(np.random.uniform(x-.1,x+.1,1)) for x in log_theta], axis=0)
        return theta_prime

    def calc_sim_score(self, params, exp_data, exp_time, ptpD=True):

        mapk_wt_data, mapk_t100a_data, map2k_wt_data, map2k_t100a_data, hog1_ramp_data, mapk_ptpD_data = exp_data
        mapk_time, mapk_time_t100a_long, mapk_ramp_time = exp_time
        mapk_time_t100a_0 = [0, 30, 60, 90, 120, 150, 180, 240, 300]

        dt = 0.1
        steps = 601
        time = np.linspace(0,dt*steps,steps)
        time_long = np.linspace(0,dt*3001,steps)

        closest_idxs_mapk = [np.abs(time - t).argmin() for t in mapk_time]
        closest_idxs_t100a_long = [np.abs(time_long - t).argmin() for t in mapk_time_t100a_0]
        closest_idxs_ramp_time = [np.abs(time - t).argmin() for t in mapk_ramp_time]

        wt_ss_inits = self.run_ss(params)
        check = self.total_protein[:-1] - wt_ss_inits[:-1]
        if (check < 0).any():
            return [((9*9)*100)**2 ]#if sims were the oppsite of the data (not possible)

        mse_total = 0
            # ptpDs
        # if ptpD:
        #     mses = np.zeros(23)
        #     ptp_doses = [0, 150000, 350000, 550000]
        #     ptpD_total_protein = total_protein[:-1] + [0]
        #     ptpD_inits = inits[:-1] + [0]
        #
        #     ptpD_ss_inits = model.run_ss(model_fxns.m, ptpD_inits, ptpD_total_protein, params)
        #     # ptpD_ss_inits = model.run_ss_ptps(model_fxns.m, inits, total_protein, params)
        #
        #     for i, (dose, data) in enumerate(zip(ptp_doses, mapk_ptpD_data), 19):
        #         odes = model.simulate_wt_experiment(model_fxns.m, ptpD_ss_inits, ptpD_total_protein, dose, params, time)
        #         # odes = model.simulate_ptpD_experiment(model_fxns.m, ptpD_ss_inits, total_protein, dose, params, time)
        #         mapk = odes[:,2]/total_protein[2]*100
        #         mses[i] = ((data - mapk[closest_idxs_mapk])**2).mean()
        #         mse_total += mses[i]
        #
        # WILDTYPE
        # Hog1
        # else:
        mses = np.zeros(19)

        for i, (dose, data) in enumerate(zip(self.hog1_doses, mapk_wt_data), 0):
            odes = self.simulate_wt_experiment(self.m, wt_ss_inits, self.total_protein, dose, params, time)#mapk_time)
            mapk = odes[:,2]/self.total_protein[2]*100
            mses[i] = ((data - mapk[closest_idxs_mapk])**2).mean()
            mse_total += mses[i]

            # Pbs2
            if dose == 150000:
                map2k = odes[:,1]/self.total_protein[1]*100
                mses[14] = ((map2k_wt_data[0] - map2k[closest_idxs_mapk])**2).mean()
                mse_total += mses[14]
            elif dose == 550000:
                map2k = odes[:,1]/self.total_protein[1]*100
                mses[15] = ((map2k_wt_data[1] - map2k[closest_idxs_mapk])**2).mean()
                mse_total += mses[15]

        # ANALOG SENSITIVE
        # Hog1
        for i, (dose, data) in enumerate(zip(self.hog1_doses, mapk_t100a_data), 7):
            if dose == 0:
                odes = self.t100a(self.m, wt_ss_inits, self.total_protein, dose, params, time_long)
                mapk = odes[:,2]/self.total_protein[2]*100
                mses[i] = ((data - mapk[closest_idxs_t100a_long])**2).mean()
                mse_total += mses[i]
            else:
                odes = self.t100a(self.m, wt_ss_inits, self.total_protein, dose, params, time)
                mapk = odes[:,2]/self.total_protein[2]*100
                mses[i] = ((data - mapk[closest_idxs_mapk])**2).mean()
                mse_total += mses[i]
                # Pbs2
                if dose == 150000:
                    map2k = odes[:,1]/self.total_protein[1]*100
                    mses[16] = ((map2k_t100a_data[0] - map2k[closest_idxs_mapk])**2).mean()
                    mse_total += mses[16]
                elif dose == 550000:
                    map2k = odes[:,1]/self.total_protein[1]*100
                    mses[17] = ((map2k_t100a_data[1] - map2k[closest_idxs_mapk])**2).mean()
                    mse_total += mses[17]

        # Hog1 ramp
        # mses[18] = 0
        # for data in hog1_ramp_data:
        #     odes = model.simulate_wt_experiment(model_fxns.m, wt_ss_inits, total_protein, 0, params, time, run_type=['ramp'])
        #     mapk = odes[:,2]/total_protein[2]*100
        #     mses[18] = ((data - mapk[closest_idxs_ramp_time])**2).mean()
    #     (mse_total/13)
        #     (mse_total/27)
        return mses

###############
#     Data    #
###############
def get_data(exp_base):

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



###############
#    Models   #
###############

def simulate_t100a_experiment_M2_(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #17
    learned_params = beta_3, 0, kb, k1, k3, k5, 0, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
    #solve odes:
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

# M2b_kb (same t100a fxn as M2a)
def M2b(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)

    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP

    dMAP3K = (((sig*k1 + kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K + MAPK*alpha)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5)*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly

    return dMAP3K, dMAP2K, dMAPK, dgly

# M2c (same t100a fxn as M2a)
def M2c(initials,t,total_protein,sig,params,run_type=None):
    # print(initials)
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)

    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1 + kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5*MAP2K + MAPK*alpha))*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)  #bug
    dgly = s7*MAPK - d8*gly
    return dMAP3K, dMAP2K, dMAPK, dgly

# M3
def M3(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK, gly, PTP = initials
    MAP3K_t, MAP2K_t, MAPK_t, _, PTP_t = total_protein
    beta_3, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = params #21

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    PTP_I = PTP_t-PTP

    dMAP3K = (((sig*k1+kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5*MAP2K)*MAPK_I)/(K_5+MAPK_I)) - ((k6 + alpha_2*PTP)*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly
    dPTP = (k9*PTP_I/(K_9+PTP_I)) - ((k10)*PTP)/(K_10+PTP)
    return dMAP3K, dMAP2K, dMAPK, dgly, dPTP

def simulate_t100a_experiment_M3(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    beta_3, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params  #21
    learned_params = beta_3, alpha_2, kb, k1, k3, k5, 0, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10

    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

def M3(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK, gly, PTP = initials
    MAP3K_t, MAP2K_t, MAPK_t, _, PTP_t = total_protein
    beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = params #21

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    PTP_I = PTP_t-PTP

    dMAP3K = (((sig*k1+kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5*MAP2K)*MAPK_I)/(K_5+MAPK_I)) - ((k6 + alpha_2*PTP)*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly
    dPTP = (k9*PTP_I/(K_9+PTP_I)) - ((k10 + alpha_1*MAPK)*PTP)/(K_10+PTP)
    return dMAP3K, dMAP2K, dMAPK, dgly, dPTP

def simulate_t100a_experiment_M3(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params #22
    learned_params = beta_3, 0, alpha_2, kb, k1, k3, k5, 0, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes
