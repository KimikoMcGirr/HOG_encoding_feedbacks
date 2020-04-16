from scipy.optimize import fsolve
from scipy.integrate import odeint

# Functions used by all models
def run_ss(m, inits, total_protein, learned_params):
    ss = fsolve(m, inits, args=(0,total_protein, 0, learned_params))
    return ss

def simulate_wt_experiment(m, inits, total_protein, sig, learned_params, time, run_type=None):
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

# Signal function
def signal_ramp_special(t_step):
    sig = 0
    if t_step >= .001:
        sig = 250000
    if t_step >= 20:
        sig = 550000

    # if t_step >= .001:
    #     sig = 100000
    # if t_step >= 5:
    #     sig = 500000
    # if t_step >= 10:
    #     sig = 550000

    # sig = 0
    # if t_step >= .001:
    #     sig = 300000
    # if t_step >= 40:
    #     sig = 450000
    # if t_step >= 45:
    #     sig = 550000

    # if t_step >= 0:
    #     sig = 300000
    return sig

tttt = 18
# def signal_ramp_special(t_step):
#     sig = 0
#     if t_step >= .001:
#         sig = 550000
#     if t_step >= 10:
#         sig = 0
#     if t_step >= 20:
#         sig = 550000
#     # if t_step >= .001:
#     #     sig = 550000
#     #
#     # sig = 0
#     # if t_step >= .001:
#     #     sig = 300000
#     # if t_step >= 40:
#     #     sig = 450000
#     # if t_step >= 45:
#     #     sig = 550000
#
#     # if t_step >= 0:
#     #     sig = 300000
#     return sig



# def signal_ramp_special(t_step):
#     sig = 0
#     if t_step >= .001:
#         sig = 300000
#     if t_step >= 20:
#         sig = 350000
#     if t_step >= 35:
#         sig = 550000
#     return sig

def get_ramp_signal(t_step, steps):
    dt = 0.1
    s = 601
    time = np.linspace(0,dt*s,s)
    if t_step == 0:
        sig = 0
        return sig
    # print(steps)
    num_steps = len(steps)
    interval = len(time)/num_steps
    i = int(np.floor(t_step/dt/interval))
    if i >= num_steps:
        sig = sum(steps)
    else:
        sig = sum(steps[0:i+1])
    # print(sig)
    return sig,

def get_manual_signal(t_step):
    sig = 0
    if t_step >= .00001:
        sig = 200000
    if t_step >= 10:
        sig = 300000
    if t_step >= 12:
        sig = 375000
    if t_step >= 25:
        sig = 500000
    if t_step >= 45:
        sig = 550000
    return sig

# def get_manual_signal(t_step):
#     sig = 0
#     if t_step >= .00001:
#         sig = 200000
#     if t_step >= 2:
#         sig = 300000
#     return sig

# def inhibit_MAPK(t_step):
# if >=0.00001:
        # data = simulate_wt_experiment(model_fxns.m, initials, params_constants, sig, params, time)

#pulse
# def get_manual_signal(t_step):
#     sig = 0
#     if t_step >= .001:
#         sig = 550000
#     if t_step >= 10:
#         sig = 0
#     if t_step >= 20:
#         sig = 550000
#     if t_step >= 30:
#         sig = 0
#     return sig

class Model():
    def __init__(self, m, t100a, nopos=None, inhib=None):
        self.m = m
        self.t100a = t100a
        self.nopos = nopos
        self.inhib = inhib
# Specific model functions
# M1
def M1(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK = initials
    MAP3K_t, MAP2K_t, MAPK_t, fb = total_protein
    beta_3, k1, k3, k5, k2, k4, k6, K_1, K_3, K_5, K_2, K_4, K_6 = params #14

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1)/(1+(MAPK/beta_3)*fb))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5)*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)

    return dMAP3K, dMAP2K, dMAPK

def simulate_t100a_experiment_M1(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    MAP3K_t, MAP2K_t, MAPK_t, fb = total_protein
    total_protein = MAP3K_t, MAP2K_t, MAPK_t, 0
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

# M2
def M2(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #16

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5)*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly

    return dMAP3K, dMAP2K, dMAPK, dgly

def simulate_t100a_experiment_M2(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    beta_3, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #16
    learned_params = beta_3, k1, k3, k5, 0, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

def M2a(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*(k1 + MAPK*alpha))/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5)*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly

    return dMAP3K, dMAP2K, dMAPK, dgly

def simulate_t100a_experiment_M2a(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    beta_3, alpha, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #17
    learned_params = beta_3, 0, k1, k3, k5, 0, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
    #solve odes:
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

# def simulate_nopos_experiment_M2c(m, inits, total_protein, sig, learned_params, time,  run_type=None):
#     beta_3, alpha, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #17
#     learned_params = beta_3, 0, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
#     #solve odes:
#     odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
#     return odes
# M2b_kb (same t100a fxn as M2a)
def M2b(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)


    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP

    dMAP3K = (((sig*k1)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
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
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)


    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5*MAP2K + MAPK*alpha))*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly
    return dMAP3K, dMAP2K, dMAPK, dgly

################################
#                              #
# MODELS WITH BASAL ACTIVATION #
#                              #
################################

def M1_kb(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK = initials
    MAP3K_t, MAP2K_t, MAPK_t, fb = total_protein
    beta_3, kb, k1, k3, k5, k2, k4, k6, K_1, K_3, K_5, K_2, K_4, K_6 = params #14

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1 + kb)/(1+(MAPK/beta_3)*fb))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5)*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)

    return dMAP3K, dMAP2K, dMAPK

def simulate_t100a_experiment_M1_kb(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    MAP3K_t, MAP2K_t, MAPK_t, fb = total_protein
    total_protein = MAP3K_t, MAP2K_t, MAPK_t, 0
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

def M1a_kb(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)
    MAP3K, MAP2K, MAPK = initials
    MAP3K_t, MAP2K_t, MAPK_t, fb = total_protein
    beta_3, alpha, kb, k1, k3, k5, k2, k4, k6, K_1, K_3, K_5, K_2, K_4, K_6 = params #14

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP

    dMAP3K = (((sig*k1 + kb +alpha*MAPK)/(1+(MAPK/beta_3)*fb))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = ((k5*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)

    return dMAP3K, dMAP2K, dMAPK

def simulate_t100a_experiment_M1a_kb(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    MAP3K_t, MAP2K_t, MAPK_t, fb = total_protein
    total_protein = MAP3K_t, MAP2K_t, MAPK_t, 0

    beta_3, alpha, kb, k1, k3, k5, k2, k4, k6, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params
    learned_params = beta_3, 0, kb, k1, k3, k5,k2, k4, k6, K_1, K_3, K_5, K_2, K_4, K_6
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

def M1b_kb(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)
    MAP3K, MAP2K, MAPK = initials
    MAP3K_t, MAP2K_t, MAPK_t, fb = total_protein
    beta_3, alpha, kb, k1, k3, k5, k2, k4, k6, K_1, K_3, K_5, K_2, K_4, K_6 = params #14

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP

    dMAP3K = (((sig*k1 + kb )/(1+(MAPK/beta_3)*fb))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K+alpha*MAPK)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = ((k5*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)

    return dMAP3K, dMAP2K, dMAPK

def M1c_kb(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)
    MAP3K, MAP2K, MAPK = initials
    MAP3K_t, MAP2K_t, MAPK_t, fb = total_protein
    beta_3, alpha, kb, k1, k3, k5, k2, k4, k6, K_1, K_3, K_5, K_2, K_4, K_6 = params #14

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP

    dMAP3K = (((sig*k1 + kb )/(1+(MAPK/beta_3)*fb))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = ((k5*MAP2K+alpha*MAPK)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)

    return dMAP3K, dMAP2K, dMAPK

def simulate_nopos_experiment_M1a_kb(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    beta_3, alpha, kb, k1, k3, k5, k2, k4, k6, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #17
    learned_params = beta_3, 0, kb, k1, k3, k5, k2, k4, k6,  K_1, K_3, K_5, K_2, K_4, K_6
    #solve odes:
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

# M2
def M2_kb(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #16

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1 + kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5)*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly

    return dMAP3K, dMAP2K, dMAPK, dgly



def simulate_t100a_experiment_M2_kb(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    beta_3, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #16
    learned_params = beta_3, kb, k1, k3, k5, 0, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

# M2a_kb
def M2a_kb(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #16

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP

    dMAP3K = (((sig*k1 + kb + MAPK*alpha)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5)*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly

    return dMAP3K, dMAP2K, dMAPK, dgly

def simulate_t100a_experiment_M2a_kb(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #17
    learned_params = beta_3, 0, kb, k1, k3, k5, 0, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
    #solve odes:
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

def simulate_nopos_experiment_M2a_kb(m, inits, total_protein, sig, learned_params, time,  run_type=None):
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #17
    learned_params = beta_3, 0, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
    #solve odes:
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

# M2b_kb (same t100a fxn as M2a)
def M2b_kb(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

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

def M2b_kb_nopos(initials,t,total_protein,sig,params,run_type=None):
    # print(initials)
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17
    alpha=0

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1 + kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K+ MAPK*alpha)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5*MAP2K ))*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)  #bug
    dgly = s7*MAPK - d8*gly
    return dMAP3K, dMAP2K, dMAPK, dgly

def M2b_kb_noneg(initials,t,total_protein,sig,params,run_type=None):
    # print(initials)
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1 + kb)/(1+(gly/beta_3)*0))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K+ MAPK*alpha)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5*MAP2K))*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)  #bug
    dgly = s7*MAPK - d8*gly
    return dMAP3K, dMAP2K, dMAPK, dgly

# M2c (same t100a fxn as M2a)
def M2c_kb(initials,t,total_protein,sig,params,run_type=None):
    # print(initials)
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

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

def M2c_kb_nopos(initials,t,total_protein,sig,params,run_type=None):
    # print(initials)
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17
    alpha=0

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1 + kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5*MAP2K + MAPK*alpha))*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)  #bug
    dgly = s7*MAPK - d8*gly
    return dMAP3K, dMAP2K, dMAPK, dgly

def M2c_kb_noneg(initials,t,total_protein,sig,params,run_type=None):
    # print(initials)
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1 + kb)/(1+(gly/beta_3)*0))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5*MAP2K + MAPK*alpha))*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)  #bug
    dgly = s7*MAPK - d8*gly
    return dMAP3K, dMAP2K, dMAPK, dgly

# M3
# def M3(initials,t,total_protein,sig,params, run_type=None):
#     if run_type:
#         if run_type[0] == 'ramp':
#             sig = signal_ramp_special(t)
#         elif run_type[0] == 'rand':
#             sig = get_ramp_signal(t, run_type[1])
#         elif run_type[0] == 'man':
#             sig = get_manual_signal(t)
#
#     MAP3K, MAP2K, MAPK, gly, PTP = initials
#     MAP3K_t, MAP2K_t, MAPK_t, _, PTP_t = total_protein
#     beta_3, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = params #21
#
#     MAP3K_I = MAP3K_t-MAP3K
#     MAP2K_I = MAP2K_t-MAP2K
#     MAPK_I = MAPK_t-MAPK
#     PTP_I = PTP_t-PTP
#
#     dMAP3K = (((sig*k1+kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
#     dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
#     dMAPK = (((k5*MAP2K)*MAPK_I)/(K_5+MAPK_I)) - ((k6 + alpha_2*PTP)*MAPK)/(K_6+MAPK)
#     dgly = s7*MAPK - d8*gly
#     dPTP = (k9*PTP_I/(K_9+PTP_I)) - ((k10)*PTP)/(K_10+PTP)
#     return dMAP3K, dMAP2K, dMAPK, dgly, dPTP
#
# def simulate_t100a_experiment_M3(m, inits, total_protein, sig, learned_params, time,  run_type=None):
#     beta_3, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params  #21
#     learned_params = beta_3, alpha_2, kb, k1, k3, k5, 0, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10
#
#     odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
#     return odes


def M1_kb_on_off(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK = initials
    MAP3K_t, MAP2K_t, MAPK_t, fb = total_protein
    beta_3, kb, k1, k3, k5, k2, k4, k6, K_1, K_3, K_5, K_2, K_4, K_6 = params #14

    if t > tttt:
        alpha = 0
        s7 = 0

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1 + kb)/(1+(MAPK/beta_3)*fb))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5)*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)

    return dMAP3K, dMAP2K, dMAPK

def M1c_kb_on_off(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)
    MAP3K, MAP2K, MAPK = initials
    MAP3K_t, MAP2K_t, MAPK_t, fb = total_protein
    beta_3, alpha, kb, k1, k3, k5, k2, k4, k6, K_1, K_3, K_5, K_2, K_4, K_6 = params #14

    if t > tttt:
        alpha = 0
        s7 = 0

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP

    dMAP3K = (((sig*k1 + kb )/(1+(MAPK/beta_3)*fb))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = ((k5*MAP2K+alpha*MAPK)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)

    return dMAP3K, dMAP2K, dMAPK

# def M2b_kb_on_off(initials,t,total_protein,sig,params, run_type=None):
#     MAP3K, MAP2K, MAPK, gly = initials
#     MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
#     beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17
#
#     if run_type:
#         if run_type[0] == 'ramp':
#             sig = signal_ramp_special(t)
#         elif run_type[0] == 'rand':
#             sig = get_ramp_signal(t, run_type[1])
#         elif run_type[0] == 'man':
#             sig = get_manual_signal(t)
#         elif run_type[0] == 'rand_off':
#             sig = get_ramp_signal(t, run_type[1], run_type[2])  ### STOP - need a way to determine when to turn kinase off
#                 # if t > run_type[2]:
#                 #     alpha = 0
#                 #     s7 = 0
#
#
#     MAP3K_I = MAP3K_t-MAP3K
#     MAP2K_I = MAP2K_t-MAP2K
#     MAPK_I = MAPK_t-MAPK
#     # PTP_I = PTP_t-PTP
#
#     dMAP3K = (((sig*k1 + kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
#     dMAP2K = (((k3*MAP3K + MAPK*alpha)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
#     dMAPK = (((k5)*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
#     dgly = s7*MAPK - d8*gly
#
#     return dMAP3K, dMAP2K, dMAPK, dgly


def M2b_kb_on_off(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)
    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17

    if t > tttt:
        alpha = 0
        s7 = 0

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP

    dMAP3K = (((sig*k1 + kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K + MAPK*alpha)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = (((k5)*MAP2K)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly

    return dMAP3K, dMAP2K, dMAPK, dgly


def M2c_kb_on_off(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)
    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17

    if t > tttt:
        alpha = 0 #alpha/4
        s7 = 0 #s7/4

        # alpha = 0
        # s7 = 0

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP
    dMAP3K = (((sig*k1 + kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = ((k5*MAP2K + MAPK*alpha)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly
    return dMAP3K, dMAP2K, dMAPK, dgly

# def M2c_kb_on_off(initials,t,total_protein,sig,params, run_type=None):
#     if run_type:
#         if run_type[0] == 'ramp':
#             sig = signal_ramp_special(t)
#         elif run_type[0] == 'rand':
#             sig = get_ramp_signal(t, run_type[1])
#         elif run_type[0] == 'man':
#             sig = get_manual_signal(t)
#     MAP3K, MAP2K, MAPK, gly = initials
#     MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
#     beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17
#
#     alpha = 0
#     if t > tttt:
#         s7 = 0
#
#     MAP3K_I = MAP3K_t-MAP3K
#     MAP2K_I = MAP2K_t-MAP2K
#     MAPK_I = MAPK_t-MAPK
#     # PTP_I = PTP_t-PTP
#     dMAP3K = (((sig*k1 + kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
#     dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
#     dMAPK = ((k5*MAP2K + MAPK*alpha)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
#     dgly = s7*MAPK - d8*gly
#     return dMAP3K, dMAP2K, dMAPK, dgly

def M2a_kb_on_off(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)
    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #17

    if t > tttt:
        alpha = 0
        s7 = 0

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP
    dMAP3K = (((sig*k1 + kb+ MAPK*alpha)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = ((k5*MAP2K )*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly
    return dMAP3K, dMAP2K, dMAPK, dgly

def simulate_inhib_experiment(m, inits, total_protein, sig, learned_params, time, run_type=None):
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

# def simulate_ptpD_experiment_M3c(m, inits, total_protein, sig, learned_params, time,  run_type=None):
#     beta_3, alpha_1,  alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params  #21
#     learned_params = beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10
#
#
#     odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
#     return odes

# M4
def M3_on_off(initials,t,total_protein,sig,params, run_type=None):
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

    if t > tttt:
        alpha_1 = 0
        s7 = 0

    # if t > 6:
    #     alpha_1 = alpha_1/2
    #     s7 = s7/2
    # if t > 7:
    #     alpha_1 = alpha_1/2
    #     s7 = s7/2
    # elif:
    #     beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params #22


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

def M3_nopos(initials, t, total_protein, sig, params, run_type=None):
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
    alpha_1 = 0

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

def M3_noPTP(initials,t,total_protein,sig,params, run_type=None):
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

    alpha_1 = 0
    alpha_2 = 0
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

# def M4_nopos(m, inits, total_protein, sig, learned_params, time, run_type=None):
#     if run_type:
#         if run_type[0] == 'ramp':
#             sig = signal_ramp_special(t)
#         elif run_type[0] == 'rand':
#             sig = get_ramp_signal(t, run_type[1])
#         elif run_type[0] == 'man':
#             sig = get_manual_signal(t)
#
#     MAP3K, MAP2K, MAPK, gly, PTP = initials
#     MAP3K_t, MAP2K_t, MAPK_t, _, PTP_t = total_protein
#     beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = params #21
#     alpha_1 = 0
#
#     MAP3K_I = MAP3K_t-MAP3K
#     MAP2K_I = MAP2K_t-MAP2K
#     MAPK_I = MAPK_t-MAPK
#     PTP_I = PTP_t-PTP
#
#     dMAP3K = (((sig*k1+kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
#     dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
#     dMAPK = (((k5*MAP2K)*MAPK_I)/(K_5+MAPK_I)) - ((k6 + alpha_2*PTP)*MAPK)/(K_6+MAPK)
#     dgly = s7*MAPK - d8*gly
#     dPTP = (k9*PTP_I/(K_9+PTP_I)) - ((k10 + alpha_1*MAPK)*PTP)/(K_10+PTP)
#     return dMAP3K, dMAP2K, dMAPK, dgly, dPTP

# def M4_new(initials,t,total_protein,sig,params, run_type=None):
    # if run_type:
    #     if run_type[0] == 'ramp':
    #         sig = signal_ramp_special(t)
    #     elif run_type[0] == 'rand':
    #         sig = get_ramp_signal(t, run_type[1])
    #     elif run_type[0] == 'man':
    #         sig = get_manual_signal(t)
    # MAP3K, MAP2K, MAPK, gly, PTP = initials
    # MAP3K_t, MAP2K_t, MAPK_t, _, PTP_t = total_protein
    # beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = params #21
    #
    # MAP3K_I = MAP3K_t-MAP3K
    # MAP2K_I = MAP2K_t-MAP2K
    # MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP
    #
    # dMAP3K = (((sig*k1+kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    # dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    # dMAPK = (((k5*MAP2K)*MAPK_I)/(K_5+MAPK_I)) - ((k6*PTP)*MAPK)/(K_6+MAPK)
    # dgly = s7*MAPK - d8*gly
    # dPTP = (k9*PTP_I/(K_9+PTP_I)) - ((k10 + alpha_1*MAPK)*PTP)/(K_10+PTP)
    #
    # return dMAP3K, dMAP2K, dMAPK, dgly, dPTP

# def simulate_ptpD_experiment(m, inits, total_protein, sig, learned_params, time, run_type=None):
    # beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params  #21
    # learned_params = beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6/alpha_2, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10
    # odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    # return odes

# def run_ss_ptps(m, inits, total_protein, learned_params):
    # beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params  #21
    # learned_params = beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6/alpha_2, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10
    # ss = fsolve(m, inits, args=(0,total_protein, 0, learned_params))
    # return ss

# def M3_kb(initials,t, total_protein, sig, params, run_type=None):
#     if run_type:
#         if run_type[0] == 'ramp':
#             sig = signal_ramp_special(t)
#         elif run_type[0] == 'rand':
#             sig = get_ramp_signal(t, run_type[1])
#         elif run_type[0] == 'man':
#             sig = get_manual_signal(t)
#     MAP3K, MAP2K, MAPK, gly, PTP = initials
#     MAP3K_t, MAP2K_t, MAPK_t, _, PTP_t = total_protein
#     beta_3, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = params #21
#
#     MAP3K_I = MAP3K_t-MAP3K
#     MAP2K_I = MAP2K_t-MAP2K
#     MAPK_I = MAPK_t-MAPK
#     PTP_I = PTP_t-PTP
#
#     dMAP3K = (((sig*k1+kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
#     dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
#     dMAPK = (((k5*MAP2K)*MAPK_I)/(K_5+MAPK_I)) - ((k6 + alpha_2*PTP)*MAPK)/(K_6+MAPK)
#     dgly = s7*MAPK - d8*gly
#     dPTP = (k9*PTP_I/(K_9+PTP_I)) - ((k10)*PTP)/(K_10+PTP)
#
#     return dMAP3K, dMAP2K, dMAPK, dgly, dPTP
#
# def simulate_t100a_experiment_M3_kb(m, inits, total_protein, sig, learned_params, time,  run_type=None):
#     beta_3, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params  #21
#     learned_params = beta_3, alpha_2, kb, k1, k3, k5, 0, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10
#     odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
#     return odes
#
# def M3c(initials,t,total_protein,sig,params, run_type=None):
#     if run_type:
#         if run_type[0] == 'ramp':
#             sig = signal_ramp_special(t)
#         elif run_type[0] == 'rand':
#             sig = get_ramp_signal(t, run_type[1])
#         elif run_type[0] == 'man':
#             sig = get_manual_signal(t)
#     MAP3K, MAP2K, MAPK, gly, PTP = initials
#     MAP3K_t, MAP2K_t, MAPK_t, _, PTP_t = total_protein
#     beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = params #21
#
#     MAP3K_I = MAP3K_t-MAP3K
#     MAP2K_I = MAP2K_t-MAP2K
#     MAPK_I = MAPK_t-MAPK
#     PTP_I = PTP_t-PTP
#
#     dMAP3K = (((sig*k1+kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
#     dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
#     dMAPK = (((k5*MAP2K+alpha_1*MAPK)*MAPK_I)/(K_5+MAPK_I)) - ((k6 + alpha_2*PTP)*MAPK)/(K_6+MAPK)
#     dgly = s7*MAPK - d8*gly
#     dPTP = (k9*PTP_I/(K_9+PTP_I)) - ((k10)*PTP)/(K_10+PTP)
#
#     return dMAP3K, dMAP2K, dMAPK, dgly, dPTP
#
#
# def simulate_t100a_experiment_M3c(m, inits, total_protein, sig, learned_params, time,  run_type=None):
#     beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params  #21
#     learned_params = beta_3, 0, alpha_2, kb, k1, k3, k5, 0, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10
#
#     odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
#     return odes
#
# def M3c_nopos(initials,t,total_protein,sig,params, run_type=None):
#     if run_type:
#         if run_type[0] == 'ramp':
#             sig = signal_ramp_special(t)
#         elif run_type[0] == 'rand':
#             sig = get_ramp_signal(t, run_type[1])
#         elif run_type[0] == 'man':
#             sig = get_manual_signal(t)
#
#     MAP3K, MAP2K, MAPK, gly, PTP = initials
#     MAP3K_t, MAP2K_t, MAPK_t, _, PTP_t = total_protein
#     beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = params #21
#
#     alpha_1 = 0
#
#     MAP3K_I = MAP3K_t-MAP3K
#     MAP2K_I = MAP2K_t-MAP2K
#     MAPK_I = MAPK_t-MAPK
#     PTP_I = PTP_t-PTP
#
#     dMAP3K = (((sig*k1+kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
#     dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
#     dMAPK = (((k5*MAP2K+alpha_1*MAPK)*MAPK_I)/(K_5+MAPK_I)) - ((k6 + alpha_2*PTP)*MAPK)/(K_6+MAPK)
#     dgly = s7*MAPK - d8*gly
#     dPTP = (k9*PTP_I/(K_9+PTP_I)) - ((k10)*PTP)/(K_10+PTP)
#
#     return dMAP3K, dMAP2K, dMAPK, dgly, dPTP

# def M5c_kb(initials,t,total_protein,sig,params, run_type=None):
#     if run_type:
#         if run_type[0] == 'ramp':
#             sig = signal_ramp_special(t)
#         elif run_type[0] == 'rand':
#             sig = get_ramp_signal(t, run_type[1])
#         elif run_type[0] == 'man':
#             sig = get_manual_signal(t)
#     MAP3K, MAP2K, MAPK, gly = initials
#     MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
#     beta_3, alpha, alpha_2, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #16
#
#     MAP3K_I = MAP3K_t-MAP3K
#     MAP2K_I = MAP2K_t-MAP2K
#     MAPK_I = MAPK_t-MAPK
#     # PTP_I = PTP_t-PTP
#
#     dMAP3K = (((sig*k1 + kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
#     dMAP2K = ((k3*MAP3K+MAP2K*alpha_2)*MAP2K_I)/(K_3+MAP2K_I) - (k4*MAP2K/(K_4+MAP2K))
#     dMAPK = ((k5*MAP2K + MAPK*alpha)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
#     dgly = s7*MAPK - d8*gly
#
#     return dMAP3K, dMAP2K, dMAPK, dgly
#
# def simulate_t100a_experiment_M5c_kb(m, inits, total_protein, sig, learned_params, time, run_type=None):
#     beta_3, alpha, alpha_2, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #16
#     learned_params = beta_3, 0, alpha_2, kb, k1, k3, k5, 0, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
#     #solve odes:
#     odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
#     return odes


## fit ptp to model 2c

def M2c_ptp(initials,t,total_protein,sig,params, run_type=None):
    if run_type:
        if run_type[0] == 'ramp':
            sig = signal_ramp_special(t)
        elif run_type[0] == 'rand':
            sig = get_ramp_signal(t, run_type[1])
        elif run_type[0] == 'man':
            sig = get_manual_signal(t)

    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6_1, k6_2, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #18

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK

    dMAP3K = (((sig*k1 + kb)/(1+gly/beta_3))*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3)*MAP3K*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = ((k5*MAP2K + MAPK*alpha)*MAPK_I)/(K_5+MAPK_I) - ((k6_1 + k6_2)*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly

    return dMAP3K, dMAP2K, dMAPK, dgly

def run_ss_M2c_ptp(m, inits, total_protein, learned_params, run_type=None):
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6_1, k6_2, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #16
    learned_params = beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6_1, 0, d8, K_1, K_3, K_5, K_2, K_4, K_6
    ss = fsolve(m, inits, args=(0,total_protein, 0, learned_params, run_type))
    return ss

def simulate_wt_experiment_M2c_ptp(m, inits, total_protein, sig, learned_params, time, run_type=None):
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6_1, k6_2, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #16
    learned_params = beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6_1, 0, d8, K_1, K_3, K_5, K_2, K_4, K_6

    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes


def simulate_t100a_experiment_M2c_ptp(m, inits, total_protein, sig, learned_params, time, run_type=None):
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6_1, k6_2, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #16
    learned_params = beta_3, 0, kb, k1, k3, k5, 0, k2, k4, k6_1, 0, d8, K_1, K_3, K_5, K_2, K_4, K_6
    #solve odes:
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

def run_ptpD_ss_M2c_ptp(m, inits, total_protein, learned_params, run_type=None):
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6_1, k6_2, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #16
    learned_params = beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, 0, k6_2, d8, K_1, K_3, K_5, K_2, K_4, K_6
    ss = fsolve(m, inits, args=(0,total_protein, 0, learned_params, run_type))
    return ss

def simulate_ptpD_experiment_M2c_ptp(m, inits, total_protein, sig, learned_params, time, run_type=None):
    beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, k6_1, k6_2, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #16
    learned_params = beta_3, alpha, kb, k1, k3, k5, s7, k2, k4, 0, k6_2, d8, K_1, K_3, K_5, K_2, K_4, K_6
    #solve odes:
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

############## RUN M3 PLOT


def run_ptpD_ss_M3_ptp(m, inits, total_protein, learned_params, run_type=None):
    beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params #22
    learned_params = beta_3, alpha_1, 0, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10
    ss = fsolve(m, inits, args=(0,total_protein, 0, learned_params))
    return ss

def simulate_ptpD_experiment_M3_ptp(m, inits, total_protein, sig, learned_params, time, run_type=None):
    beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params #22
    learned_params = beta_3, alpha_1, 0, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params))

    return odes

# def simulate_ptpD_experiment(inits, total_protein, sig, learned_params, time):
#
#     return odes
