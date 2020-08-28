from scipy.optimize import fsolve
from scipy.integrate import odeint
import numpy as np

# Functions used by all models
def run_ss(m, inits, total_protein, learned_params):
    ss = fsolve(m, inits, args=(0,total_protein, 0, learned_params))
    return ss

def simulate_wt_experiment(m, inits, total_protein, sig, learned_params, time, run_type=None):
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

def simulate_t100a_experiment(m, inits, total_protein, sig, learned_params, time, run_type=None):
    alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #16
    learned_params = 0, kb, k1, k3, k5, 0, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
    #solve odes:
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params))
    return odes

def simulate_t100a_experiment_expo(m, inits, total_protein, sig, learned_params, time):
    a, scale, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #18
    learned_params = a, scale, 0, kb, k1, k3, k5, 0, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
    #solve odes:
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params))
    return odes

def simulate_t100a_experiment_hillfxn(m, inits, total_protein, sig, learned_params, time, run_type=None):
    beta, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = learned_params #17
    learned_params = beta, 0, kb, k1, k3, k5, 0, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6
    #solve odes:
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params))
    return odes

class Model():
    def __init__(self, m, t100a=None, nopos=None, inhib=None):
        self.m = m
        self.t100a = t100a
        self.nopos = nopos
        self.inhib = inhib
# Specific model functions
# M1

def gly_lin(initials,t,total_protein,sig,params, run_type=None):
    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #16

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP

    dMAP3K = ((sig*k1 + kb - gly)*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = ((k5*MAP2K + MAPK*alpha)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly

    return dMAP3K, dMAP2K, dMAPK, dgly

def gly_expo_lin(initials,t,total_protein,sig,params, run_type=None):
    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    a, scale, alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #16

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP
    K = k1/(1+np.exp(-a*(scale*sig+kb-gly)))
    dMAP3K = (K*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = ((k5*MAP2K + MAPK*alpha)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly

    return dMAP3K, dMAP2K, dMAPK, dgly

def gly_lin_gly0(initials,t,total_protein,sig,params, run_type=None):
    MAP3K, MAP2K, MAPK, gly = initials
    MAP3K_t, MAP2K_t, MAPK_t, _ = total_protein
    alpha, kb, k1, k3, k5, s7, k2, k4, k6, d8, K_1, K_3, K_5, K_2, K_4, K_6 = params #16

    MAP3K_I = MAP3K_t-MAP3K
    MAP2K_I = MAP2K_t-MAP2K
    MAPK_I = MAPK_t-MAPK
    # PTP_I = PTP_t-PTP

    if gly < 0:
        gly = 0

    dMAP3K = ((sig*k1 + kb - gly)*MAP3K_I)/(K_1+MAP3K_I) - (k2*MAP3K/(K_2+MAP3K))
    dMAP2K = (((k3*MAP3K)*MAP2K_I)/(K_3+MAP2K_I)) - (k4*MAP2K/(K_4+MAP2K))
    dMAPK = ((k5*MAP2K + MAPK*alpha)*MAPK_I)/(K_5+MAPK_I) - (k6*MAPK)/(K_6+MAPK)
    dgly = s7*MAPK - d8*gly

    return dMAP3K, dMAP2K, dMAPK, dgly


def gly_hillfxn(initials,t,total_protein,sig,params,run_type=None):
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

def R_MM28_lin_gly0(initials,t,total_protein,sig,params,run_type=None):
    Sln1, Sho1, Hog1_AC, Hog1_AN, Hog1_IN, Glycerol = initials
    Sln1_tot, Sho1_tot, Hog1_tot, _ = total_protein
    base_osmo, k1, K1, k2, K2, k3, K3, k4, K4, k5, k6, K56, k7, K7, k8, K8, k9A, K9a, k9B, K9b, k10, K10, k11, K11, k12, k13, k14, k15 = params #18

    Hog1_IC = Hog1_tot - Hog1_AC - Hog1_AN - Hog1_IN
    Sln1_inactive = Sln1_tot - Sln1
    Sho1_inactive = Sho1_tot - Sho1

    if Glycerol < 0:
        Glycerol = 0

    dSln1 = (base_osmo + k1 * sig - Glycerol) * (Sln1_inactive) / (K1 + Sln1_inactive) - k3 * Sln1 / (K3 + Sln1)
    dSho1 = (base_osmo + k2 * sig - Glycerol) * (Sho1_inactive) / (K2 + Sho1_inactive) - k4 * Sho1 / (K4 + Sho1)
    dHog1_AC = (k5 * Sln1 + k6 * Sho1 + k14 * Hog1_AC) * Hog1_IC / (K56 + Hog1_IC) - k7 * Hog1_AC / (K7 + Hog1_AC) - (k8 + k15 * sig- Glycerol) * Hog1_AC / (K8 + Hog1_AC) + k9B * Hog1_AN / (K9b + Hog1_AN)
    dHog1_AN = (k8 + k15 * sig - Glycerol) * Hog1_AC / (K8 + Hog1_AC) - k9B * Hog1_AN / (K9b + Hog1_AN) - k10 * Hog1_AN / (K10 + Hog1_AN)
    dHog1_IN = k10 * Hog1_AN / (K10 + Hog1_AN) - k9A * Hog1_IN / (K9a + Hog1_IN) + k11 * Hog1_IC / (K11 + Hog1_IC)
    dGlycerol = k12 * Hog1_AC - k13 * Glycerol

    return dSln1, dSho1, dHog1_AC, dHog1_AN, dHog1_IN, dGlycerol
