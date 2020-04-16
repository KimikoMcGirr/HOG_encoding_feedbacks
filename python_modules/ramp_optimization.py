# from shapely.geometry import Polygon
from model import *
from scipy.optimize import fsolve
from scipy.integrate import odeint
import numpy as np
import random as rand
import matplotlib.pyplot as plt


# generate ramps
def gen_ramp():
    ramp_start = rand.randrange(50000,450000, 50000)
    ramp_end = rand.randrange(ramp_start,550000, 50000)+ramp_start
    if ramp_end > 550000:
        ramp_end = 550000
    return ramp_start, ramp_end


# def gen_step(last_step, r_start, r_end):
#     step = last_step + rand.randrange(50000, r_end-last_step, 50000)
# #     print(step)
#     if step < r_end:
#         return step
#     else:
#         gen_step(last_step, r_start, r_end)

def gen_step_size(r_start, r_end, step_num):
#     steps = [rand.randrange(r_start,r_end,50000) for x in range(step_num)]
#     if sum(steps) > r_end:
#         steps = [sum()]
    steps = [r_start]
#     total = r_end.copy()
#     print(r_start)
    diff = r_end - r_start
    for i in range(step_num-1):
#         print(steps)
        val = rand.randrange(0, diff,30000)
        steps.append(val)
        diff -= val
    return(steps)

def gen_ramp_rand():
    step_num = rand.randint(2,10)
    ramp_start, ramp_end = gen_ramp()
    return gen_step_size(ramp_start, ramp_end, step_num)


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
    return sig

# simulate ramp behavior of two models


# take average of top x %


# calculate difference


def get_odes_predict(model_fxns, ts_steps, param_sets, inits, total_protein, time, selection=''):
    traces = np.zeros([len(param_sets), len(time)])
    for i, params in enumerate(param_sets):
        ss_inits = run_ss(model_fxns.m, inits, total_protein, params)
        # if selection:
        if selection == 't100a':
            data = model_fxns.t100a(model_fxns.m, ss_inits, total_protein, 0, params, time, run_type=['rand', ts_steps])
        elif selection == 'nopos':
            data = model_fxns.nopos(model_fxns.m, ss_inits, total_protein, 0, params, time, run_type=['rand', ts_steps])
        else:
            data = simulate_wt_experiment(model_fxns.m, ss_inits, total_protein, 0, params, time, run_type=['rand', ts_steps])
        traces[i] = data[:,2]/total_protein[2]*100
    return traces

# for caculating area - old doesn't work
# def get_odes_extremes(fxn,ts_steps,param_sets,num_predict):
#     traces = []
#     for idx, param_set in enumerate(param_sets[:num_predict]):
#         data = odeint(fxn, initals, time, args=(params_constants, param_set,get_ramp_signal,[_, ts_steps, _]))
#         traces.append(data[:,2]/params_constants[2]*100)
#     traces = np.asarray(traces)
#     maxs = traces.max(axis=0)
#     mins = traces.min(axis=0)
#     return maxs,mins
# def get_overlap(maxs_1, mins_1, maxs_2, mins_2, time):
#     overlap = 0
#     t1_area = 0
#     t2_area = 0
#     for i in range(len(maxs_1)-1):
#         p1 = Polygon([(time[i],maxs_1[i]), (time[i],mins_1[i]), (time[i+1],mins_1[i+1]), (time[i+1],maxs_1[i+1])])
#         p2 = Polygon([(time[i],maxs_2[i]), (time[i],mins_2[i]), (time[i+1],mins_2[i+1]), (time[i+1],maxs_2[i+1])])
#         overlap += p2.intersection(p1).area
#         t1_area += p1.area
#         t2_area += p2.area
#     ratio = overlap/t2_area + overlap/t2_area #overlap/(t1_area+t2_area)
#     return ratio

# for calculating average
def get_average(fxn, ts_steps, param_sets, inits, total_protein, time, selection=''):
    traces = get_odes_predict(fxn, ts_steps, param_sets, inits, total_protein, time, selection)
    return np.average(traces, axis=0)

def get_mse(ave1, ave2):
    return sum((ave1 - ave2)**2)

def get_differentiating_ramp(num_runs, time,
                            model_params1, model_fxns1, inits1, total_protein1,
                            model_params2, model_fxns2, inits2, total_protein2, selection=''):
    mse_gen = []
    averages = []
    for i in range(num_runs):
        ts_steps = gen_ramp_rand()
#         print(params_sets[0])
        # for model_params in
        ave1 = get_average(model_fxns1, ts_steps, model_params1, inits1, total_protein1, time, selection)
        ave2 = get_average(model_fxns2, ts_steps, model_params2, inits2, total_protein2, time, selection)

        mse = get_mse(ave1, ave2)#/[get_ramp_signal(x, ts_steps) for x in time]
        # print(mse)
        if i == 0:
            ramp_vals = [mse,ts_steps]
            averages = [ave1, ave2]
        else:
            if ramp_vals[0] < mse:
                # print(mse)
                ramp_vals = [mse,ts_steps]
                averages = [ave1, ave2]
        mse_gen.append(mse)#ramp_vals[0])

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))
    ax1.plot(range(num_runs), mse_gen, linewidth=4)
    ax2.plot(time,[get_ramp_signal(x, ramp_vals[1]) for x in time], linewidth=3)
        # if i % 10 == 0:
        #     plt.clf()
        #     plt.plot(time,[get_ramp_signal(x, ramp_vals[1],_)/1000 for x in time])
        #     plt.ylim(0,600)
        #     plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/ramp_gen/"+str(i)+".jpeg",bbox_inches='tight',dpi=150)
    print(ramp_vals)
    return ramp_vals, averages, mse_gen
