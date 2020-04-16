import numpy as np
from model import *

def save_sim_data(model_fxns, top_params, top, params_constants, initials, doses, time, param,
                    ss=False, path=None, sim_name=None, t100a=False, ramp=False):
    sims_to_save = []
    if ramp:
        for params in top_params[:top]:
            if ss:
                ss_data = run_ss(model_fxns.m, initials, params_constants, params)
                data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, 0, params, time, run_type=['ramp'])
            else:
                data = simulate_wt_experiment(model_fxns.m, initials, params_constants, 0, params, time, run_type=['ramp'])
            active = data[:,param]/params_constants[param]*100
            sims_to_save.append(active)
    else:
        for signal in doses:
            for params in top_params[:top]:
                if ss:
                    ss_data = run_ss(model_fxns.m, initials, params_constants, params)
                    if t100a:
                        data = model_fxns.t100a(model_fxns.m, ss_data, params_constants, signal, params, time)
                    else:
                        data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, signal, params, time)
                else:
                    if t100a:
                        data = model_fxns.t100a(model_fxns.m, initials, params_constants, signal, params, time)
                    else:
                        data = simulate_wt_experiment(model_fxns.m, initials, params_constants, signal, params, time)
                active = data[:,param]/params_constants[param]*100
                sims_to_save.append(active)
    if path and sim_name:
        np.savetxt(path + sim_name + "_time.csv", time, delimiter=",")
        np.savetxt(path + sim_name + "_simulations.csv", np.asarray(sims_to_save).T, delimiter=",")
    else:
        return sims_to_save

def save_man_data(model_fxns, top_params, top, params_constants, initials, time, param,
                    ss=False, path=None, sim_name=None, t100a=False):
    sims_to_save = []
    for params in top_params[:top]:
        if ss:
            ss_data = run_ss(model_fxns.m, initials, params_constants, params)
            data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, 0, params, time, run_type=['man'])
        else:
            data = simulate_wt_experiment(model_fxns.m, initials, params_constants, 0, params, time, run_type=['man'])
        active = data[:,param]/params_constants[param]*100
        sims_to_save.append(active)
    if path and sim_name:
        np.savetxt(path + sim_name + "_time.csv", time, delimiter=",")
        np.savetxt(path + sim_name + "_KCl.csv", [get_manual_signal(x)/1000 for x in time], delimiter=",")
        np.savetxt(path + sim_name + "_simulations.csv", np.asarray(sims_to_save).T, delimiter=",")
    else:
        return sims_to_save
