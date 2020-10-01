import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
import seaborn as sns
from model import *
import pandas as pd
import warnings

MAPK_palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#8ace88', 250000:'#5ab769', 350000: '#319a50', 450000:'#117b38', 550000:'#005723'}
MAP2K_palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#84bcdb', 250000:'#5ab769', 350000: '#319a50', 450000:'#117b38', 550000:'#084082'}
MAP3K_palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#fb7d5d', 250000:'#5ab769', 350000: '#319a50', 450000:'#117b38', 550000:'#840711'}
osmo_palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#adabd2', 250000:'#5ab769', 350000: '#7566ae', 450000:'#117b38', 550000:'#491285'}
ptp_palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#D9A673', 250000:'#B3804D', 350000: '#8C5926', 450000:'#794613', 550000:'#663300'}
pinks = {0:'#323232', 50000:'#f6ab83', 150000:'#f06043', 250000:'#B3804D', 350000: '#8C5926', 450000:'#794613', 550000:'#cb1b50'}

MAPK_palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#5ab769', 250000:'#5ab769', 350000: '#319a50', 450000:'#117b38', 550000:'#005723'}
MAP2K_palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#84bcdb', 250000:'#5ab769', 350000: '#319a50', 450000:'#117b38', 550000:'#084082'}
MAP3K_palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#fb7d5d', 250000:'#5ab769', 350000: '#319a50', 450000:'#117b38', 550000:'#840711'}
osmo_palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#adabd2', 250000:'#5ab769', 350000: '#7566ae', 450000:'#117b38', 550000:'#491285'}
ptp_palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#D9A673', 250000:'#B3804D', 350000: '#8C5926', 450000:'#794613', 550000:'#663300'}
pinks = {0:'#323232', 50000:'#f6ab83', 150000:'#f06043', 250000:'#B3804D', 350000: '#8C5926', 450000:'#794613', 550000:'#cb1b50'}


palettes = {0:MAP3K_palette,
           1:MAP2K_palette,
           2: MAPK_palette,
           3:osmo_palette,
           4:ptp_palette }

x_labels = {0: '% active MAP3K',
          1: '% pp Pbs2',
          2: '% pp Hog1',
          3: 'Osmolyte (uM)',
          4: '% active phosphatase'}

param_colors = {'M1': ['#be202e','#606060', '#851747','#33669a','#05582d', '#851747','#33669a','#05582d', '#851747','#33669a','#05582d', '#851747','#33669a','#05582d'],
                'M1_kb': ['#be202e','#606060','#258b44', '#851747','#33669a','#05582d', '#851747','#33669a','#05582d', '#851747','#33669a','#05582d', '#851747','#33669a','#05582d'],
                'M2': ['#be202e','#258b44', '#851747','#33669a','#05582d', '#5f3c99', '#851747','#33669a','#05582d', '#5f3c99', '#851747','#33669a','#05582d', '#851747','#33669a','#05582d'],
                'M2a': ['#be202e','#258b44', '#606060','#851747','#33669a','#05582d', '#5f3c99', '#851747','#33669a','#05582d', '#5f3c99', '#851747','#33669a','#05582d', '#851747','#33669a','#05582d'],
                'M3': ['#be202e','#258b44', '#606060','#851747','#33669a','#05582d', '#5f3c99', '#8C5926','#851747','#33669a','#05582d', '#5f3c99', '#8C5926','#851747','#33669a','#05582d', '#8C5926','#851747','#33669a','#05582d', '#8C5926'],
                'M3c': ['#be202e', '#be202e','#258b44','#606060', '#606060','#851747','#33669a','#05582d', '#5f3c99', '#8C5926','#851747','#33669a','#05582d', '#5f3c99', '#8C5926','#851747','#33669a','#05582d', '#8C5926','#851747','#33669a','#05582d', '#8C5926'],
                'M4': ['#be202e', '#258b44','#258b44', '#606060','#851747','#33669a','#05582d', '#5f3c99', '#8C5926','#851747','#33669a','#05582d', '#5f3c99', '#8C5926','#851747','#33669a','#05582d', '#8C5926','#851747','#33669a','#05582d', '#8C5926']

          }

def simulate_ptpD_experiment(m, inits, total_protein, sig, learned_params, time, run_type=None):
    beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params  #21
    learned_params = beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6/alpha_2, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10
    odes = odeint(m, inits, time, args=(total_protein, sig, learned_params, run_type))
    return odes

def run_ss_ptps(m, inits, total_protein, learned_params):
    beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10 = learned_params  #21
    learned_params = beta_3, alpha_1, alpha_2, kb, k1, k3, k5, s7, k9, k2, k4, k6/alpha_2, d8, k10, K_1, K_3, K_5, K_9, K_2, K_4, K_6, K_10
    ss = fsolve(m, inits, args=(0,total_protein, 0, learned_params))
    return ss

def plt_param_behaviors(model_fxns, top_params, plt_top, params_constants, initials,  doses, time, param,
                        mapk_wt_data=None, mapk_t100a_data=None, mapk_time=None, ptpD=False, ss=True, plt_bad=0,
                        save_fig=''):
    plt.clf()
    font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 20}
    plt.rc('font', **font)

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,4))

    # plot 1
#     title_text = 'Gen ' + str(gen) + ' best fits to WT'
    # title_text = 'Wild-type Simulations'
    # ax1.set_title(title_text, fontsize=20)
    # ax1.set_xlabel('Time (min)', fontsize=16)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    palette = palettes.get(param)
    # ax1.set_ylabel(x_labels.get(param), fontsize=24)
    # ax1.set_yticks(np.arange(0, 101, step=25))
    # ax1.set_xticks(np.arange(0, 61, step=15))

    if mapk_wt_data:
        # tdoses = [0, 150000, 350000, 550000]
        for sig, wt_data in zip(doses, mapk_wt_data):
            # ax1.plot(mapk_time, wt_data, 'o', markersize=10, color=palette.get(sig), label = str(int(sig/1000))+'mM KCl')
            ax1.plot(mapk_time, wt_data, 'o', markersize=10, color=palette.get(sig), label = str(int(sig/1000))+'mM KCl')

#     ax1.legend()

    # plot 2
#     title_text = 'Gen ' + str(gen) +  ' best fits to T100A + inhib'
    # title_text = 'Inhibited MAPK Simulations'#'Best fits to kinase dead mutant dose data'
    # ax2.set_title(title_text, fontsize=20)
    # ax2.set_xlabel('Time (min)', fontsize=16)
    # ax2.set_yticks(np.arange(0, 101, step=25))
    # ax2.set_xticks(np.arange(0, 61, step=15))
    # ax2.set_ylabel(x_labels.get(param), fontsize=16)

    if mapk_t100a_data:
        for sig, t100a_data in zip(doses, mapk_t100a_data):
#             print(len(t100a_data))
#             if sig == 0:
#                 continue
#                 print(mapk_time_t100a_long)
#                 ax2.plot(mapk_time_t100a_long, t100a_data, '^', mew=2, markersize=10, color=palette.get(sig))
#             else:
            ax2.plot(mapk_time, t100a_data, '^', mew=2, markersize=10, color=palette.get(sig))


    if param == 3:
#         ax1.set_ylim(0,150)
        dt = 0.1
        steps = 601
        time = np.linspace(0,dt*steps,steps)
        # ax1.set_yscale("log", nonposy='clip')
        # ax2.set_yscale("log", nonposy='clip')
    else:
        ax1.set_ylim(-5,105)
        ax2.set_ylim(-5,105)

#     if params_constants[-1] == 0:
#         ax1.set_ylim(50,105)
    if ptpD:
        # for params in top_params[:plt_top]:
        #     ptpD_total_protein = params_constants[:-2]+[550000*2, 0]
        #     ptpD_inits = initials[:-1]+[0]
        #
        #     ptpD_ss_inits = run_ss(model_fxns.m, ptpD_inits, ptpD_total_protein, params)
        #     # ptpD_ss_inits = run_ss_ptps(model_fxns.m, initials, params_constants, params)
        #
        #     # check = params_constants - ptpD_ss_inits
        #     # if (check < 0).any():
        #         # return ((63+69+18*2+27)*100)**2
        #     # else:
        #         # mse_total = 0
        #     ptp_doses = [0, 150000, 350000, 550000]
        #     plt.rc('xtick', labelsize=20)
        #     plt.rc('ytick', labelsize=20)
        #     # for dose, data in zip(doses, mapk_ptp_data):
        #     for sig in doses:
        #
        #         odes = simulate_wt_experiment(model_fxns.m, ptpD_ss_inits, ptpD_total_protein, sig, params, time)
        #         # odes = simulate_ptpD_experiment(model_fxns.m, ptpD_ss_inits, params_constants, sig, params, time)
        #
        #         active = odes[:,param]/params_constants[param]*100
        #         ax1.plot(time, active, color=pinks.get(sig), linewidth=3, alpha=.75)
        #         # data = model_fxns.t100a(model_fxns.m, ptpD_ss_inits, params_constants, sig, params, time)
        #         data = model_fxns.t100a(model_fxns.m, ptpD_ss_inits, params_constants, sig, params, time)
        #         active = data[:,param]/params_constants[param]*100
        #         ax2.plot(time, active, '--', color=palette.get(sig))
        #         ax1.set_ylim(50,100)
        #         ax1.set_yticks(np.arange(50, 101, step=10))
        for params in top_params[:plt_top]:
            # ptpD_total_protein = params_constants[:-2]+[550000*2, 0]
            # ptpD_inits = initials[:-1]+[0]

            # ptpD_ss_inits = run_ptpD_ss_M2c_ptp(model_fxns.m, initials, params_constants, params)
            ptpD_ss_inits = run_ptpD_ss_M3_ptp(model_fxns.m, initials, params_constants, params)

            # ptpD_ss_inits = run_ss_ptps(model_fxns.m, initials, params_constants, params)

            check = params_constants - ptpD_ss_inits
            if (check < 0).any():
                continue
            # else:
                # mse_total = 0
            # ptp_doses = [0, 150000, 350000, 550000]
            ptp_doses = [0]

            plt.rc('xtick', labelsize=20)
            plt.rc('ytick', labelsize=20)
            # for dose, data in zip(doses, mapk_ptp_data):
            for sig in doses:
                # odes = simulate_ptpD_experiment_M2c_ptp(model_fxns.m, ptpD_ss_inits, params_constants, sig, params, time)
                odes = simulate_ptpD_experiment_M3_ptp(model_fxns.m, ptpD_ss_inits, params_constants, sig, params, time)

                # odes = simulate_ptpD_experiment(model_fxns.m, ptpD_ss_inits, params_constants, sig, params, time)

                active = odes[:,param]/params_constants[param]*100
                ax1.plot(time, active, color=pinks.get(sig), linewidth=3, alpha=.75)
                # data = model_fxns.t100a(model_fxns.m, ptpD_ss_inits, params_constants, sig, params, time)
                data = model_fxns.t100a(model_fxns.m, ptpD_ss_inits, params_constants, sig, params, time)
                active = data[:,param]/params_constants[param]*100
                ax2.plot(time, active, '--', color=palette.get(sig))
                # ax1.set_ylim(50,100)
                ax1.set_yticks(np.arange(50, 101, step=10))

    else:
        # ax1.plot(time, np.zeros(len(time)), color=palette.get(0))
        # ax2.plot(time, np.zeros(len(time)), '--', color=palette.get(0))
        # print(doses)
        for sig in doses:
            for params in top_params[:plt_top]:
                    if ss:
                        ss_data = run_ss(model_fxns.m, initials, params_constants, params)
                        # print(ss_data)
                        data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, sig, params, time)
                        # print('hi')
                    else:
                        data = simulate_wt_experiment(model_fxns.m, initials, params_constants, sig, params, time)
                    if param == 3:
                        active = [x/ss_data[3] for x in data[:,param]]
                    else:
                        active = data[:,param]/params_constants[param]*100
                    ax1.plot(time, active, color=palette.get(sig))

                    if ss:
                        ss_data = run_ss(model_fxns.m, initials, params_constants, params)
                        data = model_fxns.t100a(model_fxns.m, ss_data, params_constants, sig, params, time)
                    else:
                        data = model_fxns.t100a(model_fxns.m, initials, params_constants, sig, params, time)
                    if param == 3:
                        active = [x/ss_data[3] for x in data[:,param]]
                    else:
                        active = data[:,param]/params_constants[param]*100
                    ax2.plot(time, active, '--', color=palette.get(sig))

#     ax1.legend(bbox_to_anchor=[1, 0.5], loc='center left')
    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=1)
    ax2.grid(color='grey', linestyle='-', axis='y', linewidth=1)
    if plt_bad:
        plt_thresh_behavior(model_fxns, top_params, plt_bad, params_constants, initials,  doses, time, param, ax1, ax2)
    if save_fig:
        plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/simulations/"+save_fig+".png",
        dpi=300,bbox_inches='tight')
    plt.show()

def plt_ramp_behaviors(model_fxns, top_params, plt_top, params_constants, initials, time, param,
                        ss = False, hog1_ramp_data=None, mapk_ramp_time=None,
                        save_fig=''):
    fig, (ax1) = plt.subplots(1, 1, figsize=(9,4))

    ax1.plot(mapk_ramp_time, hog1_ramp_data[0], 'o', markersize=10, color='Black')

    colors = sns.color_palette("bone")
    pal2 = sns.set_palette(colors)
    ax1.set_ylabel('% pp Hog1', fontsize=16)
    ax1.set_xlabel('Time (min)', fontsize=16)
    ax1.set_yticks(np.arange(0, 101, step=25))
    ax1.set_xticks(np.arange(0, 61, step=15))

    for params in top_params[:plt_top]:
#         for sig in doses:
            if ss:
                ss_data = run_ss(model_fxns.m, initials, params_constants, params)
                data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, 0, params, time, run_type=['ramp'])
            else:
                data = simulate_wt_experiment(model_fxns.m, initials, params_constants, 0, params, time, run_type=['ramp'])
            active = data[:,param]/params_constants[param]*100
            ax1.plot(time, active)
    if save_fig:
        plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/simulations/"+save_fig+".png",
        dpi=300,bbox_inches='tight')
    plt.show()



def plt_man_behaviors(model_fxns, top_params, plt_top, params_constants, initials, time, param,
                        ss=False, save_fig=''):
    fig, (ax1) = plt.subplots(1, 1, figsize=(9,4))

    colors = sns.color_palette("Set2", 10)
    pal2 = sns.set_palette(colors)
    ax1.set_ylabel('% pp Hog1', fontsize=16)
    ax1.set_xlabel('Time (min)', fontsize=16)
    ax1.set_ylim(0,100)

    for params in top_params[:plt_top]:
#         for sig in doses:
            if ss:
                ss_data = run_ss(model_fxns.m, initials, params_constants, params)
                data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, 0, params, time, run_type=['man'])
            else:
                data = simulate_wt_experiment(model_fxns.m, initials, params_constants, 0, params, time, run_type=['man'])
            active = data[:,param]/params_constants[param]*100
            ax1.plot(time, active)
    ax2 = ax1.twinx()
    ax2.plot(time, [get_manual_signal(x)/1000 for x in time], color='black', linewidth=1)
    if save_fig:
        plt.savefig("../Figures/"+save_fig+".png",
        dpi=300,bbox_inches='tight')
    plt.show()

def plt_man_behaviors_t100a(model_fxns, top_params, plt_top, params_constants, initials, time, param,
                        save_fig=''):
    fig, (ax1) = plt.subplots(1, 1, figsize=(9,4))

    colors = sns.color_palette("Set2", 10)
    pal2 = sns.set_palette(colors)
    ax1.set_ylabel('% pp Hog1', fontsize=16)
    ax1.set_xlabel('Time (min)', fontsize=16)
    ax1.set_ylim(0,100)

    for params in top_params[:plt_top]:
#         for sig in doses:
            ss_data = run_ss(model_fxns.m, initials, params_constants, params)
            data = model_fxns.t100a(model_fxns.m, ss_data, params_constants, 0, params, time, run_type=['man'])
            active = data[:,param]/params_constants[param]*100
            ax1.plot(time, active)
    ax2 = ax1.twinx()
    ax2.plot(time, [get_manual_signal(x)/1000 for x in time], color='black', linewidth=1)
    if save_fig:
        plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/simulations/"+save_fig+".png",
        dpi=300,bbox_inches='tight')
    plt.show()



def plt_sses_gen(gen, mses, idx_top, save_fig=''):
    plt.clf()
    fig, (ax3) = plt.subplots(1, 1, figsize=(9,4))
    colors2 = sns.color_palette("Greys", 20)[10:]
    pal2 = sns.set_palette(colors2)
    ax3.set_xlabel('Generation', fontsize=20)
    for mse in mses[:idx_top]:
        # ax3.semilogy([x for x in range(gen)], mse[idx][:gen])
        ax3.plot([x for x in range(gen)], mse[:gen])
    ax3.yaxis.grid(True)
    ax3.set_ylabel('SSE', fontsize=20)
    ax3.set_xlim([0,gen])
    # ax3.set_ylim(1000,5000)
    # ax3.set_ylim([3.5,6])

    if save_fig:
        plt.savefig("../Figures/"+save_fig+".png",
        dpi=300,bbox_inches='tight')
    plt.show()

def plt_param_ranges(labelnames, m_name, dims, param_data, single_theta=pd.Series(),num=0, plot = 'swarm',save_fig=''):
    # fig, (ax1) = plt.subplots(1, 1, figsize=(3,3))

    if save_fig:
        fig, (ax1) = plt.subplots(1, 1, figsize=(2.25,2))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(18,4))


    # positive 3db7d6
    # negative ee9537
    # hog1 ? 38bf85 or 94ddaa

    # colors = ['#606060','#606060','#606060',
    #           '#851747','#33669a','#228a44', '#5f3c99', '#a97c50',
    #           '#851747','#33669a','#228a44', '#5f3c99', '#a97c50',
    #           '#851747','#33669a','#228a44', '#a97c50',
    #           '#851747','#33669a','#228a44', '#a97c50']
    pal = sns.set_palette(param_colors.get(m_name))

    # Hide the right and top spiness
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    major_ticks = np.arange(-4, 9, 2)
    ax1.set_yticks(major_ticks)

    # plt.bar(range(0,len(labelnames)),height=dims[0],bottom=dims[1],align='center',tick_label=labelnames, color='#dcdcdc',alpha = 0.8)
    with sns.axes_style("whitegrid"):
        if plot == 'swarm':
            ax1 = sns.swarmplot(x='param',y='vals', data = param_data, size=4) #size 3
        elif plot == 'violin':
            ax1 = sns.violinplot(x='param',y='vals', data = param_data, size=5,   bw=.5, width=2) #size 3 , inner='stick' cut=0,

        ax1.set_xticklabels(labelnames,rotation=90)
        plt.xlabel('Parameters', fontsize=20, fontweight='medium')
        ax1.set_ylabel('')

    plt.grid(color='#606060', which='major', axis='y', linestyle='solid')
    if single_theta.any:
        single_theta = pd.DataFrame(single_theta.loc[num])
        single_theta['param']=single_theta.index
        single_theta.melt(var_name='param', value_name='vals')
        ax1 = sns.swarmplot(x='param', y=num, data = single_theta, color = 'black', size=6)
    # a=ax1.get_yticks().tolist()
    # y_names=['$10^{-6}$','$10^{-4}$','$10^{-2}$','$1$','$10^{2}$','$10^{4}$','$10^{6}$',]
    y_names=['$10^{-4}$','$10^{-2}$','$1$','$10^{2}$','$10^{4}$','$10^{6}$','$10^{8}$']
    ax1.set_yticklabels(y_names)

    if save_fig:
        plt.savefig(save_fig,
        dpi=300,bbox_inches='tight')
    plt.show()

def plt_mses_swarm(mses, save_fig=''): # hog1=True, t100a=False, pbs2=False, pbs2_t100a=False, ramp=False, ptp23D=False,
    mses = mses.apply(np.log10).melt(var_name='Dataset', value_name='MSEs')
    # mses = mses
    fig, (ax1) = plt.subplots(1, 1, figsize=(9,4))
    sns.swarmplot(x='Dataset', y="MSEs",
                            data=mses, palette="muted") #
    # ax1.set_xticklabels(['WT Hog1', 'Hog1-as Hog1', 'WT Pbs2', 'Hog1-as Pbs2', 'Ramp', 'Ptp2/3$\Delta$'],rotation=45)
    plt.show()

def plt_mses(mses, datasets, size, ptpD=True,save_fig=''): # hog1=True, t100a=False, pbs2=False, pbs2_t100a=False, ramp=False, ptp23D=False,
    # mses = mses.apply(np.log10).melt(var_name='Dataset', value_name='$\Sigma$ MSEs')
    hog1_doses = [0, 50000,150000,250000,350000,450000,550000]
    ptp_doses = [0, 150000,550000]
    mses_means = mses.mean()
    r1 = 'Hog1 WT'

    barWidth = .8
    r2 = 'Hog1-as'
    r3 = 'Pbs2 WT'
    r4 = 'Pbs2-as'
    r5 = 'Ramp'
    # r6 = 'Ptp2/3$\Delta$'

    fig, ax1 = plt.subplots(1, 1, figsize=size)


    if datasets[0]:
        ax1.bar(r1, mses_means[0], width=barWidth, edgecolor='white', color=MAPK_palette.get(hog1_doses[0]))#, label='Hog1'
        for i, dose in zip(range(1,7),hog1_doses[1:]) :
            ax1.bar(r1, mses_means[i], bottom=sum(mses_means[0:i]),  width=barWidth, edgecolor='white', color=MAPK_palette.get(dose))

    if datasets[1]:
        ax1.bar(r2, mses_means[7], width=barWidth, edgecolor='white', color=MAPK_palette.get(hog1_doses[0]))
        for i, dose in zip(range(1,7),hog1_doses[1:]) :
            ax1.bar(r2, mses_means[i+7], bottom=sum(mses_means[7:i+7]),  width=barWidth, edgecolor='white', color=MAPK_palette.get(dose))

    if datasets[2]:
        ax1.bar(r3, mses_means[14], width=barWidth, edgecolor='white', color=MAP2K_palette.get(150000))#, label='Hog1'
        ax1.bar(r3, mses_means[15], bottom=mses_means[14],  width=barWidth, edgecolor='white', color=MAP2K_palette.get(550000))

    if datasets[3]:
        ax1.bar(r4, mses_means[16], width=barWidth, edgecolor='white', color=MAP2K_palette.get(150000))#, label='Hog1'
        ax1.bar(r4, mses_means[17], bottom=mses_means[16],  width=barWidth, edgecolor='white', color=MAP2K_palette.get(550000))

    if datasets[4]:
        ax1.bar(r5, mses_means[18], width=barWidth, edgecolor='white', color='#008080')#, label='Hog1'

    if ptpD:
        if datasets[5]:
            ax1.bar(r6, mses_means[19], width=barWidth, edgecolor='white', color=ptp_palette.get(ptp_doses[0]))
            for i, dose in zip(range(20,22), ptp_doses[1:]):
                ax1.bar(r6, mses_means[i], bottom=mses_means[i-1],  width=barWidth, edgecolor='white',color=ptp_palette.get(dose))
        # ax1.set_xticklabels([r6],rotation=45)
        ax1.set_xticklabels([r1,r2,r3,r4,r5],rotation=45)



    # if predicted:
    #     plt.yticks([])
    #     ax1.set_yscale("log", nonposy='clip')
    plt.tight_layout()


    # plt.savefig(figure_dir+'fig3_D'+".pdf",dpi=200)
    # ax1.legend()
    plt.ylabel("MSE")
    # plt.xticklabels(labelnames,rotation=45)
    print(sum(mses_means[:-1]))
    if save_fig:
        plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/simulations/"+save_fig+".png",
        dpi=300,bbox_inches='tight')
    plt.show()

def plt_t100a_long(model_fxns, top_params, plt_top, params_constants, initials, t100a_data, mapk_time, save_fig=''):
    plt.clf()
    fig, (ax1) = plt.subplots(1, 1, figsize=(9,4))

    ax1.set_ylabel('% pp Hog1', fontsize=16)
    ax1.set_xlabel('Time (min)', fontsize=16)
    title_text = 'Inhibited MAPK Simulations'
    ax1.set_title(title_text, fontsize=20)
    ax1.set_xlabel('Time (min)', fontsize=16)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    ax1.plot(mapk_time, t100a_data, '^', mew=2, markersize=10, color=MAPK_palette.get(0))

    dt = 0.1
    steps = 3001
    time = np.linspace(0,dt*steps,steps)
    for params in top_params[:plt_top]:
        ss_data = run_ss(model_fxns.m, initials, params_constants, params)
        data = model_fxns.t100a(model_fxns.m, ss_data, params_constants, 0, params, time)
        active = data[:,2]/params_constants[2]*100
        ax1.plot(time, active, '--', color=MAPK_palette.get(0))

    plt.ylim(0,100)

    if save_fig:
        plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/simulations/"+save_fig+".png",
        dpi=300,bbox_inches='tight')
    plt.show()

# def plt_nopos(model_fxns, top_params, plt_top, params_constants, initials, exp_data, mapk_time, sig, save_fig=''):
#     plt.clf()
#     fig, (ax1) = plt.subplots(1, 1, figsize=(8,3))
#
#     ax1.set_ylabel('% pp Hog1', fontsize=16)
#     ax1.set_xlabel('Time (min)', fontsize=16)
#     title_text = 'No positive feedback at ' + str(int(sig/1000)) + 'mM KCl'
#
#     ax1.set_title(title_text, fontsize=20)
#     ax1.set_xlabel('Time (min)', fontsize=16)
#     plt.rc('xtick', labelsize=14)
#     plt.rc('ytick', labelsize=14)
#
#     ax1.plot(mapk_time, exp_data, '^', mew=2, markersize=10, color='black')
#     ax1.yaxis.grid(True)
#     dt = 0.1
#     steps = 301
#     time = np.linspace(0,dt*steps,steps)
#     for params in top_params[:plt_top]:
#         ss_data = run_ss(model_fxns.m, initials, params_constants, params)
#
#         # data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, 0, params, time)
#         # active = data[:,2]/params_constants[2]*100
#         # ax1.plot(time, active, color = 'black')
#
#         data = model_fxns.nopos(model_fxns.m, ss_data, params_constants, sig, params, time)
#         active = data[:,2]/params_constants[2]*100
#         ax1.plot(time, active, '--')
#
#     plt.ylim(-5,105)
#     plt.xlim(0,30)


    # if save_fig:
    #     plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/simulations/"+save_fig+".png",
    #     dpi=300,bbox_inches='tight')
    # plt.show()

# def plt_mses(mses, save_fig=''): # hog1=True, t100a=False, pbs2=False, pbs2_t100a=False, ramp=False, ptp23D=False,
#     mses = mses.apply(np.log10).melt(var_name='Dataset', value_name='$\Sigma$ MSEs')
#     # mses = mses
#     fig, (ax1) = plt.subplots(1, 1, figsize=(9,4))
#     ax1 = sns.violinplot(x='Dataset', y="$\Sigma$ MSEs",
#                             data=mses, palette="muted") #
#     ax1.set_xticklabels(['WT Hog1', 'Hog1-as Hog1', 'WT Pbs2', 'Hog1-as Pbs2', 'Ramp', 'Ptp2/3$\Delta$'],rotation=45)
#     plt.show()


def plt_corr(labelnames, df_top_params, save_fig=''):
    df_top_params.columns = labelnames
    df_top_params.head()
    cmap = sns.cm.rocket

    fig, (ax1) = plt.subplots(1, 1, figsize=(8,6))

    ax1 = sns.heatmap(df_top_params.corr(), cmap = cmap,  vmin=-.1, vmax=1.1) #annot=True,  fmt='d'
    plt.yticks(rotation=0)
    ax1.set_xticklabels(labelnames,rotation=90)
    ax1.tick_params(labelsize=12)
    # ax1.set_xlabel('Parameters', fontweight='medium', fontsize=18)
    # sns.set(font_scale=1.4)
    if save_fig:
        plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/simulations/"+save_fig+".png",
            dpi=300,bbox_inches='tight')
    s = df_top_params.corr().unstack()
    so = s.sort_values(kind="quicksort")
    print(so[-10-len(labelnames):-len(labelnames)])
    plt.show()

def plt_rand_behaviors(model_fxns, top_params, plt_top, params_constants, initials, time,
                            ramp_vals, average, selection='',
                            save_fig=''):
    fig, (ax1) = plt.subplots(1, 1, figsize=(9,4))

    colors = sns.color_palette("Set2", 10)
    pal2 = sns.set_palette(colors)
    ax1.set_ylabel('% pp Hog1', fontsize=16)
    ax1.set_xlabel('Time (min)', fontsize=16)

    for params in top_params[:plt_top]:
        ss_data = run_ss(model_fxns.m, initials, params_constants, params)
        if selection == 't100a':
            data = model_fxns.t100a(model_fxns.m, ss_data, params_constants, 0, params, time, run_type=['rand', ramp_vals[1]])
        elif selection == 'nopos':
            data = model_fxns.nopos(model_fxns.m, ss_data, params_constants, 0, params, time, run_type=['rand', ramp_vals[1]])
        else:
            data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, 0, params, time, run_type=['rand', ramp_vals[1]])
        active = data[:,2]/params_constants[2]*100
        ax1.plot(time, active)
    ax1.plot(time, average, linewidth=4, color='black')
    if save_fig:
        plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/simulations/"+save_fig+".png",
        dpi=300,bbox_inches='tight')
    ax1.set_ylim(0,100)
    plt.show()

def plt_deriv(mses, zoom, elim, deriv=1, window=False):
    a1 = np.sort(mses)[:elim]
    if window:
        a2 = np.concatenate([a1[window:], a1[-window:]])
    else:
        a2 = np.concatenate([a1[1:], [a1[-1]]])
    z = abs(a2-a1)
    if deriv == 2:
       b2 = np.concatenate([z[1:], [z[-1]]])
       z = b2-z
       # print(z)
    thresh = max(z)*0.01
    idx_thresh = np.argwhere(z<thresh)[0]
    fig, (ax1) = plt.subplots(1, 1, figsize=(9,4))
    plt.axhline(y=thresh, color='black', linewidth=3, label='1% of max')
    # plt.axvline(x=idx_thresh, color='black', linewidth=3)
    ax1.plot(range(zoom),z[:zoom], linewidth=3, color='grey')
    # ax1.plot(range(len(z)),z, 'o')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('First derivative, win='+str(window))
    plt.legend()
    return thresh, idx_thresh

def plt_idx_vs_sse(sses, zoom, sorting=True,idx_thresh=False,data_thresh=False):
    if sorting:
        sses = np.sort(sses)[:zoom]
    else:
        sses = np.array(sses)[:zoom]
    fig, ax1 = plt.subplots(1, 1, figsize=(9,4))
    plt.plot(range(len(sses)),sses, 'o', markersize=3, color='grey')
    plt.xlabel('Index')
    # plt.ylim(10000,100000)
    # plt.ylabel('$\sum MSE$')
    if idx_thresh:
        plt.axvline(x=idx_thresh, color='black', linewidth=3, label='thresh')
    if data_thresh:
        # plt.axhline(y=data_thresh, color='black', linewidth=3, label='thresh')
        _idx = [i for i, val in enumerate(sses) if val<data_thresh]
        # print(_idx)
        # plt.
        plt.plot(_idx, sses[_idx], 'o', markersize=3, color='pink')
        return _idx

    plt.legend()
    # ax1.semilogy(range(len(mses)),mses)

def plt_pdf(mses, zoom, idx_thresh=False):
    # mses = np.sort(mses)[:zoom]
    fig, ax1 = plt.subplots(1, 1, figsize=(9,4))
    plt.plot(range(len(mses)),mses, 'o', markersize=3, color='grey')
    plt.xlabel('Index')
    # plt.ylabel('$\sum MSE$')
    if idx_thresh:
        plt.axvline(x=idx_thresh, color='black', linewidth=3, label='1% max slope of the derivative')
    plt.legend()
# def plt_thresh_behavior(model_fxns, top_params, plt_num, params_constants, initials,  doses, time, param, ax1, ax2):
#     bad_run = top_params[plt_num]
#     for sig in doses:
#         ss_data = run_ss(model_fxns.m, initials, params_constants, bad_run)
#         odes = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, sig, bad_run, time)
#         active = odes[:,param]/params_constants[param]*100
#         ax1.plot(time, active, color='#dc2924', alpha=.75)
#         data = model_fxns.t100a(model_fxns.m, ss_data, params_constants, sig, bad_run, time)
#         active = data[:,param]/params_constants[param]*100
#         ax2.plot(time, active, '--', color='#dc2924', alpha=.75)


# crappy slow f u seaborn
# def simdata_to_df(model_fxns, top_params, params_constants, initials, time, param,
#                         ss = False):
#     colnames = ['idx', 'time', 'value']
#     sims = pd.DataFrame(columns = colnames)
#     for idx, params in enumerate(top_params):
#         if ss:
#             ss_data = run_ss(model_fxns.m, initials, params_constants, params)
#             data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, 0, params, time, run_type=['ramp'])
#         else:
#             data = simulate_wt_experiment(model_fxns.m, initials, params_constants, 0, params, time, run_type=['ramp'])
#         active = data[:,2]/params_constants[2]*100
#         for t, point in enumerate(active):
#             if t % 5 == 0:
#                 sims = sims.append({'idx': idx, 'time': t/10, 'value': point}, ignore_index=True)
#         if idx % int(len(top_params)*.1) == 0:
#             print(str(int(idx/len(top_params)*100)) + "% complete.")
#     return sims


def fit_data_to_list2(model_fxns, top_params, params_constants, initials, time, param, dose, t100a=False, ptpD=False,
                        ss = False):
    sims = []
    for idx, params in enumerate(top_params):
        ss_data = run_ss(model_fxns.m, initials, params_constants, params)
        data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, dose, params, time)
        active = data[:,param]/params_constants[param]*100

        sims.append(active)

        up = params.copy()
        up[0] *= 3
        ss_data = run_ss(model_fxns.m, initials, params_constants, up)
        data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, dose, up, time)
        active = data[:,param]/params_constants[param]*100

        sims.append(active)

        down = params.copy()
        down[0] /= 2
        ss_data = run_ss(model_fxns.m, initials, params_constants, down)
        data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, dose, down, time)
        active = data[:,param]/params_constants[param]*100

        sims.append(active)

    print('Dose: ' + str(dose) + ' complete.')
    return sims

def fit_data_to_list(model_fxns, top_params, params_constants, initials, time, param, dose, t100a=False, ptpD=False,
                        ss = False):
        sims = []
        for idx, params in enumerate(top_params):
            if ss:
                ss_data = model_fxns.ss(model_fxns.m, initials, params_constants, params)
                if t100a:
                    data = model_fxns.t100a(model_fxns.m, ss_data, params_constants, dose, params, time)
                else:
                    if ptpD:
                        # ptp_doses = [0, 150000, 350000, 550000]
                        ptpD_total_protein = params_constants[:-1] + [0]
                        ptpD_inits = initials[:-1] + [0]

                        ptpD_ss_inits = model_fxns.ss(model_fxns.m, ptpD_inits, ptpD_total_protein, params)
                        # ptpD_ss_inits = model.run_ss_ptps(model_fxns.m, inits, total_protein, params)

                        data = model_fxns.wt(model_fxns.m, ptpD_ss_inits, ptpD_total_protein, dose, params, time)
                    else:
                        data = model_fxns.wt(model_fxns.m, ss_data, params_constants, dose, params, time)
            else:
                if t100a:
                    data = model_fxns.t100a(model_fxns.m, initials, params_constants, dose, params, time)
                else:
                    data = model_fxns.wt(model_fxns.m, initials, params_constants, dose, params, time)
            if param == 3:
                # max_data = np.max(data)
                # min_data = np.min(data)
                # active = [(1-0)*(x-min_data)/(max_data-min_data)+0 for x in data]
                # active = [(x-data[:,param][0])/data[:,param][0] for x in data[:,param]]
                # active = [(x-ss_data[3])/ss_data[3] for x in data[:,param]]
                active = [x/ss_data[3] for x in data[:,param]]

            else:
                active = data[:,param]/params_constants[param]*100

            sims.append(active)
        print('Dose: ' + str(dose) + ' complete.')
        return sims

def nopos_to_list(model_fxns, top_params, params_constants, initials, time, param, dose, t100a=False, ptpD=False, ss=True):
        sims = []
        for idx, params in enumerate(top_params):
            data = model_fxns.nopos(model_fxns.m, initials, params_constants, dose, params, time)
            if param == 3:
                # max_data = np.max(data)
                # min_data = np.min(data)
                # active = [(1-0)*(x-min_data)/(max_data-min_data)+0 for x in data]
                # active = [(x-data[:,param][0])/data[:,param][0] for x in data[:,param]]
                # active = [(x-ss_data[3])/ss_data[3] for x in data[:,param]]
                active = [x/ss_data[3] for x in data[:,param]]

            else:
                active = data[:,param]/params_constants[param]*100

            sims.append(active)
        print('Dose: ' + str(dose) + ' complete.')
        return sims

def noneg_to_list(model_fxns, top_params, params_constants, initials, time, param, dose, t100a=False, ptpD=False, ss=True):
        sims = []
        for idx, params in enumerate(top_params):
            data = model_fxns.noneg(model_fxns.m, initials, params_constants, dose, params, time)
            if param == 3:
                # max_data = np.max(data)
                # min_data = np.min(data)
                # active = [(1-0)*(x-min_data)/(max_data-min_data)+0 for x in data]
                # active = [(x-data[:,param][0])/data[:,param][0] for x in data[:,param]]
                # active = [(x-ss_data[3])/ss_data[3] for x in data[:,param]]
                active = [x/ss_data[3] for x in data[:,param]]

            else:
                active = data[:,param]/params_constants[param]*100

            sims.append(active)
        print('Dose: ' + str(dose) + ' complete.')
        return sims


def plt_param_cis(model_fxns, top_params, params_constants, initials,  doses, time, param,
                        exp_data=None, exp_time=None, ss=False, t100a=False, ptpD=False, ci= 95,
                        save_fig='', save_as='.pdf'):
    plt.clf()
    font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 16}
    plt.rc('font', **font)

    if save_fig:
        fig, (ax1) = plt.subplots(1, 1, figsize=(2.25,2))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))

    plt.rc('ytick', labelsize=16)
    plt.rc('xtick', labelsize=16)

    palette = palettes.get(param)

    if param == 3:
        dt = 0.1
        steps = 1801
        time = np.linspace(0,dt*steps,steps)
    # elif doses == [0]:
    #     dt = 0.1
    #     steps = 3001
    #     time = np.linspace(0,dt*steps,steps)




    # dashes = None
    # if t100a:
    dashes= (2,2)

    # pred_col = ['#8da0ca', '#68c3a4', '#f37e80']
    # pred_col = ['#68c3a4']

    # for pred, sig in zip(pred_col, doses):
    for sig in doses:
        if t100a:
            if ptpD:
                # ax1.lines[0].set_linestyle("--")
                sims = fit_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=True, ptpD=True, ss=ss)
            else:
                sims = fit_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=True, ptpD=False, ss=ss)
            # if params == 3:
            #     ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
            # else:
            # ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
        else:
            if ptpD:
                sims = fit_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=False, ptpD=True, ss=ss)
                # ax1.lines[0].set_linestyle("--")
            else:
                sims = fit_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=False, ptpD=False, ss=ss)
        # if sig == 350000:
            # ax1 = sns.tsplot(sims, time,  ci = ci, color='#a97c50', dashes=dashes)
            # ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
            # else:
        # print(sims[0])
        # 1/0
        # sims = [ s for s in np.array(sims)]
        # sims = list(filter(lambda i: i.all() <101, sims))
        # print(sims[1])
        # print(len(sims))
        # test = []
        # for e, s in enumerate(sims):
        #     for i in s:
        #         if i > 100:
        #             print(e)
        #             continue
        # test = np.array([s for s in sims])
        # print(len(test))
        # # test[np.all(test, axis=0) < 100]
        # print(len(test))
        # print(type(test))
        # print(type(test[0]))
        # print(np.all())
        # test = []
        # for s in sims:
        #     if sum(s) > 60001:
        #         continue
        #     else:
        #         test.append(s)


        # sims = sims[:3]+sims[4:56]+sims[57:79]+sims[80:97]+sims[98:]
        # print(sims)
        # ax1 = sns.tsplot(sims[0], time,  ci = ci, color='#228833', linewidth=3)#, dashes=dashes)
        # ax1 = sns.tsplot(sims[1], time,  ci = ci, color='#4477aa', linewidth=3)
        # ax1 = sns.tsplot(sims[2], time,  ci = ci, color='#aa3377', linewidth=3)

        # ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes, err_style='unit_traces')
        ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)#, err_style='unit_traces')
        # ax1 = sns.tsplot(test, time,  ci = ci, color=pred, dashes=dashes)



    mstyles = {0: ['full'],
                1: ['^', 'o'],
                2: ['D', 'o', '^', 'v', 's', 'D', 'o'],
                3: ['full'],
                4: ['full']}
    fstyles = {0: ['full'],
                1: ['full', 'bottom'],
                2: ['none', 'none'],
                3: ['full'],
                4: ['full']}

    if exp_data:
        if t100a:
            for sig, data in zip(doses, exp_data): #, mstyles.get(param)
                mark = 'o'
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 250000:
                    mark = 'o'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 450000:
                    mark = 'D'
                elif sig == 550000:
                    mark = 's'
                ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='none', color=palette.get(sig), mec='black', label = str(int(sig/1000)))
        else:
            for sig, data in zip(doses, exp_data):
                mark = 'o'
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 250000:
                    mark = 'o'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 450000:
                    mark = 'D'
                elif sig == 550000:
                    mark = 's'
                # if sig == 350000:
                    # ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='full', color='#a97c50', mec='black', label = str(int(sig/1000)))#, fillstyle=fill, linestyle="None"), palette.get(sig), mec='black' (outside edge)
                # else:
                ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='full', color=palette.get(sig), mec='black', label = str(int(sig/1000)))#, fillstyle=fill, linestyle="None"), palette.get(sig), mec='black' (outside edge)






    # if plt_bad:
        # plt_thresh_behavior(model_fxns, top_params, plt_bad, params_constants, initials,  doses, time, param, ax1, ax2)

    if param == 3:
        ax1.set_xticks(np.arange(0, 181, step=60))
        ax1.set_xticks(np.arange(0, 61, step=15))
        # ax1.set_xticks(np.arange(0, 21, step=5))
        ax1.set_yticks(np.arange(0, 5, step=1))
        # ax1.set_ylim(1,2.75)
        ax1.set_ylim(1,4.5)
        ax1.set_xlim(-1,61)



    elif doses == [0]:
        ax1.set_ylim(-5,105)
        ax1.set_yticks(np.arange(0, 101, step=25))
        ax1.set_xticks(np.arange(0, 2401, step=60))
        ax1.set_ylim(-5,105)
        ax1.set_xlim(-5,245)
        plt.xticks(rotation=45)
        # ax1.set_xticklabels(rotation=30)
    else:
        ax1.set_yticks(np.arange(0, 101, step=25))
        # ax1.set_xticks(np.arange(0, 181, step=60))
        ax1.set_xticks(np.arange(0, 61, step=15))
        # ax1.set_xticks(np.arange(0, 21, step=5))
        ax1.set_ylim(-5,105)
        # ax1.set_xlim(-2,182)
        ax1.set_xlim(-2,62)
        # ax1.set_xlim(-1,21)



    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # ax1.legend(bbox_to_anchor=[1, 0.5], loc='best')

    plt.xlabel('Time (min)')
    # plt.ylabel('% ppHog1')

    if save_fig:
        plt.savefig(save_fig+save_as, dpi=300,bbox_inches='tight')

    plt.show()

def plt_param_cis_justin_predictions(model_fxns, top_params, params_constants, initials,  doses, time, param,
                        exp_data=None, exp_time=None, ss=False, t100a=False, ptpD=False, ci= 95,
                        save_fig='', save_as='.pdf'):
    plt.clf()
    font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 16}
    plt.rc('font', **font)

    if save_fig:
        fig, (ax1) = plt.subplots(1, 1, figsize=(2.25,2))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))

    plt.rc('ytick', labelsize=16)
    plt.rc('xtick', labelsize=16)

    palette = palettes.get(param)


    if param == 3:
        dt = 0.1
        steps = 1801
        time = np.linspace(0,dt*steps,steps)
    # elif doses == [0]:
    #     dt = 0.1
    #     steps = 3001
    #     time = np.linspace(0,dt*steps,steps)




    # dashes = None
    # if t100a:
    dashes= (2,2)

    for sig, color in zip(doses, ['#8da0cb','#66c2a4','#ff8080']):
        if t100a:
            sims = fit_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=True, ptpD=False, ss=ss)
            # if params == 3:
            #     ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
            # else:
            # ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
        else:
            sims = fit_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=False, ptpD=False, ss=ss)
        # if sig == 350000:
            # ax1 = sns.tsplot(sims, time,  ci = ci, color='#a97c50', dashes=dashes)
            # ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
            # else:
        ax1 = sns.tsplot(sims, time,  ci = ci, color=color, dashes=dashes)

    mstyles = {0: ['full'],
                1: ['^', 'o'],
                2: ['D', 'o', '^', 'v', 's', 'D', 'o'],
                3: ['full'],
                4: ['full']}
    fstyles = {0: ['full'],
                1: ['full', 'bottom'],
                2: ['none', 'none'],
                3: ['full'],
                4: ['full']}

    if exp_data:
        if t100a:
            for sig, data, color in zip(doses, exp_data, ['#8da0cb','#66c2a4','#ff8080']): #, mstyles.get(param)
                mark = 'o'
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 250000:
                    mark = 'o'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 450000:
                    mark = 'D'
                elif sig == 550000:
                    mark = 's'
                ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='none', color=color, mec='black', label = str(int(sig/1000)))
        else:
            for sig, data, color in zip(doses, exp_data, ['#8da0cb','#66c2a4','#ff8080']):
                mark = 'o'
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 250000:
                    mark = 'o'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 450000:
                    mark = 'D'
                elif sig == 550000:
                    mark = 's'
                # if sig == 350000:
                    # ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='full', color='#a97c50', mec='black', label = str(int(sig/1000)))#, fillstyle=fill, linestyle="None"), palette.get(sig), mec='black' (outside edge)
                # else:
                ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='full', color=color, mec='black', label = str(int(sig/1000)))#, fillstyle=fill, linestyle="None"), palette.get(sig), mec='black' (outside edge)





    # if plt_bad:
        # plt_thresh_behavior(model_fxns, top_params, plt_bad, params_constants, initials,  doses, time, param, ax1, ax2)

    if param == 3:
        ax1.set_xticks(np.arange(0, 181, step=60))
        ax1.set_yticks(np.arange(0, 5, step=1))
        ax1.set_ylim(1,4.25)

    elif doses == [0]:
        ax1.set_ylim(-5,105)
        ax1.set_yticks(np.arange(0, 101, step=25))
        ax1.set_xticks(np.arange(0, 301, step=60))
        ax1.set_ylim(-5,105)
        ax1.set_xlim(-5,305)
        plt.xticks(rotation=45)
        # ax1.set_xticklabels(rotation=30)
    else:
        ax1.set_yticks(np.arange(0, 101, step=25))
        ax1.set_xticks(np.arange(0, 61, step=15))
        ax1.set_ylim(-5,105)
        ax1.set_xlim(-2,62)

    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend(bbox_to_anchor=[1, 0.5], loc='best')

    if save_fig:
        plt.savefig(save_fig+save_as, dpi=300,bbox_inches='tight')

    plt.show()
    
def plt_param_cis_neg_predictions(model_fxns, top_params, params_constants, initials,  doses, time, param,
                        exp_data=None, exp_time=None, ss=False, t100a=False, ptpD=False, ci= 95,
                        save_fig='', save_as='.pdf'):
    plt.clf()
    font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 16}
    plt.rc('font', **font)

    if save_fig:
        fig, (ax1) = plt.subplots(1, 1, figsize=(2.25,2))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))

    plt.rc('ytick', labelsize=16)
    plt.rc('xtick', labelsize=16)

    palette = palettes.get(param)


    if param == 3:
        dt = 0.1
        steps = 1801
        time = np.linspace(0,dt*steps,steps)
    # elif doses == [0]:
    #     dt = 0.1
    #     steps = 3001
    #     time = np.linspace(0,dt*steps,steps)




    # dashes = None
    # if t100a:
    dashes= (2,2)

    for sig, color in zip(doses, ['#8da0cb','#66c2a4','#ff8080']):
        if t100a:
            sims = noneg_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=True, ptpD=False, ss=ss)
            # if params == 3:
            #     ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
            # else:
            # ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
        else:
            sims = noneg_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=False, ptpD=False, ss=ss)
        # if sig == 350000:
            # ax1 = sns.tsplot(sims, time,  ci = ci, color='#a97c50', dashes=dashes)
            # ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
            # else:
        ax1 = sns.tsplot(sims, time,  ci = ci, color=color, dashes=dashes)

    mstyles = {0: ['full'],
                1: ['^', 'o'],
                2: ['D', 'o', '^', 'v', 's', 'D', 'o'],
                3: ['full'],
                4: ['full']}
    fstyles = {0: ['full'],
                1: ['full', 'bottom'],
                2: ['none', 'none'],
                3: ['full'],
                4: ['full']}

    if exp_data:
        if t100a:
            for sig, data, color in zip(doses, exp_data, ['#8da0cb','#66c2a4','#ff8080']): #, mstyles.get(param)
                mark = 'o'
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 250000:
                    mark = 'o'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 450000:
                    mark = 'D'
                elif sig == 550000:
                    mark = 's'
                ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='none', color=color, mec='black', label = str(int(sig/1000)))
        else:
            for sig, data, color in zip(doses, exp_data, ['#8da0cb','#66c2a4','#ff8080']):
                mark = 'o'
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 250000:
                    mark = 'o'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 450000:
                    mark = 'D'
                elif sig == 550000:
                    mark = 's'
                # if sig == 350000:
                    # ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='full', color='#a97c50', mec='black', label = str(int(sig/1000)))#, fillstyle=fill, linestyle="None"), palette.get(sig), mec='black' (outside edge)
                # else:
                ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='full', color=color, mec='black', label = str(int(sig/1000)))#, fillstyle=fill, linestyle="None"), palette.get(sig), mec='black' (outside edge)





    # if plt_bad:
        # plt_thresh_behavior(model_fxns, top_params, plt_bad, params_constants, initials,  doses, time, param, ax1, ax2)

    if param == 3:
        ax1.set_xticks(np.arange(0, 181, step=60))
        ax1.set_yticks(np.arange(0, 5, step=1))
        ax1.set_ylim(1,4.25)

    elif doses == [0]:
        ax1.set_ylim(-5,105)
        ax1.set_yticks(np.arange(0, 101, step=25))
        ax1.set_xticks(np.arange(0, 301, step=60))
        ax1.set_ylim(-5,105)
        ax1.set_xlim(-5,305)
        plt.xticks(rotation=45)
        # ax1.set_xticklabels(rotation=30)
    else:
        ax1.set_yticks(np.arange(0, 101, step=25))
        ax1.set_xticks(np.arange(0, 61, step=15))
        ax1.set_ylim(-5,105)
        ax1.set_xlim(-2,62)

    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend(bbox_to_anchor=[1, 0.5], loc='best')

    if save_fig:
        plt.savefig(save_fig+save_as, dpi=300,bbox_inches='tight')

    plt.show()

def plt_param_cis_pos_predictions(model_fxns, top_params, params_constants, initials,  doses, time, param,
                        exp_data=None, exp_time=None, ss=False, t100a=False, ptpD=False, ci= 95,
                        save_fig='', save_as='.pdf'):
    plt.clf()
    font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 16}
    plt.rc('font', **font)

    if save_fig:
        fig, (ax1) = plt.subplots(1, 1, figsize=(2.25,2))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))

    plt.rc('ytick', labelsize=16)
    plt.rc('xtick', labelsize=16)

    palette = palettes.get(param)


    if param == 3:
        dt = 0.1
        steps = 1801
        time = np.linspace(0,dt*steps,steps)
    # elif doses == [0]:
    #     dt = 0.1
    #     steps = 3001
    #     time = np.linspace(0,dt*steps,steps)




    # dashes = None
    # if t100a:
    dashes= (2,2)

    for sig, color in zip(doses, ['#8da0cb','#66c2a4','#ff8080']):
        if t100a:
            sims = nopos_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=True, ptpD=False, ss=ss)
            # if params == 3:
            #     ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
            # else:
            # ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
        else:
            sims = nopos_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=False, ptpD=False, ss=ss)
        # if sig == 350000:
            # ax1 = sns.tsplot(sims, time,  ci = ci, color='#a97c50', dashes=dashes)
            # ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)
            # else:
        ax1 = sns.tsplot(sims, time,  ci = ci, color=color, dashes=dashes)

    mstyles = {0: ['full'],
                1: ['^', 'o'],
                2: ['D', 'o', '^', 'v', 's', 'D', 'o'],
                3: ['full'],
                4: ['full']}
    fstyles = {0: ['full'],
                1: ['full', 'bottom'],
                2: ['none', 'none'],
                3: ['full'],
                4: ['full']}

    if exp_data:
        if t100a:
            for sig, data, color in zip(doses, exp_data, ['#8da0cb','#66c2a4','#ff8080']): #, mstyles.get(param)
                mark = 'o'
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 250000:
                    mark = 'o'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 450000:
                    mark = 'D'
                elif sig == 550000:
                    mark = 's'
                ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='none', color=color, mec='black', label = str(int(sig/1000)))
        else:
            for sig, data, color in zip(doses, exp_data, ['#8da0cb','#66c2a4','#ff8080']):
                mark = 'o'
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 250000:
                    mark = 'o'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 450000:
                    mark = 'D'
                elif sig == 550000:
                    mark = 's'
                # if sig == 350000:
                    # ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='full', color='#a97c50', mec='black', label = str(int(sig/1000)))#, fillstyle=fill, linestyle="None"), palette.get(sig), mec='black' (outside edge)
                # else:
                ax1.plot(exp_time, data, marker=mark, markersize=6, linestyle="-", fillstyle='full', color=color, mec='black', label = str(int(sig/1000)))#, fillstyle=fill, linestyle="None"), palette.get(sig), mec='black' (outside edge)





    # if plt_bad:
        # plt_thresh_behavior(model_fxns, top_params, plt_bad, params_constants, initials,  doses, time, param, ax1, ax2)

    if param == 3:
        ax1.set_xticks(np.arange(0, 181, step=60))
        ax1.set_yticks(np.arange(0, 5, step=1))
        ax1.set_ylim(1,4.25)

    elif doses == [0]:
        ax1.set_ylim(-5,105)
        ax1.set_yticks(np.arange(0, 101, step=25))
        ax1.set_xticks(np.arange(0, 301, step=60))
        ax1.set_ylim(-5,105)
        ax1.set_xlim(-5,305)
        plt.xticks(rotation=45)
        # ax1.set_xticklabels(rotation=30)
    else:
        ax1.set_yticks(np.arange(0, 101, step=25))
        ax1.set_xticks(np.arange(0, 61, step=15))
        ax1.set_ylim(-5,105)
        ax1.set_xlim(-2,62)

    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend(bbox_to_anchor=[1, 0.5], loc='best')

    if save_fig:
        plt.savefig(save_fig+save_as, dpi=300,bbox_inches='tight')

    plt.show()

def simdata_to_list(model_fxns, top_params, params_constants, initials, time, param,
                        ss = False):
    sims = []
    for idx, params in enumerate(top_params):
        try:
            if ss:
                ss_data = fsolve(model_fxns.m, initials, args=(0,params_constants, 0, params))
                data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, 350000, params, time, run_type=['ramp'])
                active = data[:,param]/params_constants[param]*100
                sims.append(active)
            else:
                data = simulate_wt_experiment(model_fxns.m, initials, params_constants, 0, params, time, run_type=['man'])
        except RuntimeWarning:
            print("Runtimewarning at idx: "+str(idx))

        if len(top_params) > 1:
            if idx % int(len(top_params)*.1) == 0:
                print(str(int(idx/len(top_params)*100)) + "% complete.")

    return sims

def plt_ramp(num, rampnum=None, save_fig=''):

    if save_fig:
        fig, (ax1) = plt.subplots(1, 1, figsize=(2.25,1))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # for i in []:
        # ax1 = sns.tsplot(sims[:,:num], time[:num],  ci = ci)
        # print(len(time[0]))
    # 1/0
    # time = [0,.001,4.99,5,9.99,10,60]
    # salt = [0, 150, 150, 500,500, 550, 550]
    dt = 0.1
    steps = 1801
    time = np.linspace(0,dt*steps,steps)

    ramp = np.zeros(len(time))
    # ramp[:] = 550
    # ramp[:100] = 500
    # ramp[:50] = 100
    # ramp[0] = 0

    # ramp[:] = 550
    # ramp[:200] = 250
    # # ramp[:50] = 100
    # ramp[0] = 0

    # ax1.plot([0,.001,19.99,20,60],[0, 250, 250,550,550], linewidth=2)
    # for num in range(12):


    if rampnum == 0:
        colorl = '#ff8080'
        ramp[:] = 550
        ramp[:100] = 500
        ramp[:50] = 100
        ramp[0] = 0
    if rampnum == 1:
        colorl = '#66c2a4'
        ramp[:] = 550
        ramp[:200] = 250
        # ramp[:50] = 100
        ramp[0] = 0
    if rampnum == 2:
        colorl = '#8da0cb'
        ax1.axvline(x=18, dashes=(2,2), color='red', linewidth=2)
        ramp[:] = 550
        ramp[:200] = 250
        # ramp[:50] = 100
        ramp[0] = 0

    if rampnum == 3:
        colorl = '#58595b'
        # ax1.axvline(x=18, dashes=(2,2), color='red', linewidth=2)
        ramp[:] = 350
        ramp[0] = 0

    ax1.plot(time[:num],ramp[:num], color=colorl, linewidth=2)

    ax1.set_ylim(-5,605)
    ax1.set_xlim(-2,61)

    ax1.set_yticks(np.arange(0, 601, step=200))
    ax1.set_xticks(np.arange(0, 61, step=15))


    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    if save_fig:
        plt.savefig(save_fig+str(num)+'.jpeg', dpi=300,bbox_inches='tight')
    # plt.show()

def plt_ramp_cis(model_fxns, thetas, params_constants, inits, time, num, ramp=None, ramp_data=None, ramp_time=None, ci=95,
                        save_fig=''):

    _sims = simdata_to_list(model_fxns, thetas, params_constants, inits, time, 2, ss = True)
    _sims = np.array(_sims)

    if save_fig:
        fig, (ax1) = plt.subplots(1, 1, figsize=(2.25,2))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))

    # fig.clear()
    if ramp == 0:
        colorl = '#66c2a4'
    if ramp == 1:
        colorl = '#ff8080'
    if ramp == 2:
        colorl = '#8da0cb'
        ax1.axvline(x=18, dashes=(2,2), color='red', linewidth=2)

    # dashes=(2,2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # for i in []:
        ax1 = sns.tsplot(_sims, time)#, err_style='unit_traces') #dashes
        # print(len(time[0]))
    # 1/0

    if ramp_data:
        ax1.plot(ramp_time, ramp_data[0], '*', markersize=6, color='Black')




    ax1.set_yticks(np.arange(0, 101, step=25))
    ax1.set_xticks(np.arange(0, 61, step=15))
    ax1.set_ylim(-5,105)
    ax1.set_xlim(-2,62)

    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=.5)
    ax1.grid(color='white', linestyle='-', axis='x', linewidth=.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    if save_fig:
        plt.savefig(save_fig, dpi=300,bbox_inches='tight')
    # plt.show()

def plt_nopos_cis(sims, time, num, hog1_data=None, mapk_time=None, ci=95,
                        save_fig=''):

    sims = np.array(sims)
    if save_fig:
        fig, (ax1) = plt.subplots(1, 1, figsize=(2.25,2))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))

    # fig.clear()

    colorl = '#309a4f'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # for i in []:
        ax1 = sns.tsplot(sims[:,:num], time[:num],  ci = ci, color=colorl)
        # print(len(time[0]))
    # 1/0

    if hog1_data:
        ax1.plot(mapk_time, hog1_data[0], 'o', markersize=10, color='Black')




    ax1.set_yticks(np.arange(0, 101, step=25))
    ax1.set_xticks(np.arange(0, 61, step=15))
    ax1.set_ylim(-5,105)
    ax1.set_xlim(-2,62)

    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=.5)
    ax1.grid(color='white', linestyle='-', axis='x', linewidth=.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    if save_fig:
        plt.savefig(save_fig, dpi=300,bbox_inches='tight')


def inhibdata_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, run_type=None,
                        ss = False):
    sims = []
    for idx, params in enumerate(top_params):
        if ss:
            ss_data = run_ss(model_fxns.inhib, initials, params_constants, params)
            data = simulate_inhib_experiment(model_fxns.inhib, ss_data, params_constants, sig, params, time, run_type)
        else:
            data = simulate_inhib_experiment(model_fxns.inhib, initials, params_constants, sig, params, time, run_type)
        if param ==3:
            active = [x/ss_data[3] for x in data[:,param]]
        else:
            active = data[:,param]/params_constants[param]*100
        sims.append(active)
        if idx % int(len(top_params)*.1) == 0:
            print(str(int(idx/len(top_params)*100)) + "% complete.")
    return sims

def plt_inhib_cis(sims, time, param,  ci=95, save_fig=''):

    fig, (ax1) = plt.subplots(1, 1, figsize=(5,4))

    ax1 = sns.tsplot(sims, time,  ci = ci)

    if param ==3:
        ax1.set_yticks(np.arange(0, 8, step=2))
        ax1.set_ylim(0,8)
    else:
        ax1.set_yticks(np.arange(0, 101, step=25))
        ax1.set_ylim(-5,105)
    ax1.set_xlim(-2,61)
    ax1.set_xticks(np.arange(0, 61, step=15))

    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=1)

    # ax1.plot(mapk_ramp_time, hog1_ramp_data[0], 'o', markersize=10, color='Black')
    # ax1 = sns.lineplot(x = "time", y = "value", data=sims, ci = ci)


    # ax2.plot(mapk_ramp_time, hog1_ramp_data[0], 'o', markersize=10, color='Black')
    # ax2 = sns.lineplot(x = "time", y = "value", data=sims, ci = 'sd')

    if save_fig:
        plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/simulations/"+save_fig+".png",
        dpi=300,bbox_inches='tight')

# def nopos_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=False, ptpD=False, ss=ss):
#     # warnings.filterwarnings('error')
#     sims = []
#     for idx, params in enumerate(top_params):
#         if ss:
#             # with np.seterr(divide='raise'):
#             try:
#                 # ss_data = fsolve(model_fxns.nopos, initials, args=(0,params_constants, 0, params))
#                 ss_data = run_ss(model_fxns.nopos, initials, params_constants, params)
#                 data = simulate_wt_experiment(model_fxns.nopos, ss_data, params_constants, 350000, params, time, run_type)
#                 active = data[:,2]/params_constants[2]*100
#                 sims.append(active)
#             except RuntimeWarning:
#                 print("Runtimewarning at idx: "+str(idx))
#                 # continue
#
#             # else:
#
#         # else:
#         #     data = simulate_wt_experiment(model_fxns.nopos, initials, params_constants, 350000, params, time)
#
#         if idx % int(len(top_params)*.1) == 0:
#             print(str(int(idx/len(top_params)*100)) + "% complete.")
#     return sims

def M4_noptps_list(model_fxns, top_params, params_constants, initials, time, param,
                        ss = False):
    # warnings.filterwarnings('error')
    sims = []
    for idx, params in enumerate(top_params):
        if ss:
            # with np.seterr(divide='raise'):
            try:
                ss_data = fsolve(model_fxns.m, initials[:-1]+[0], args=(0,params_constants[:-1]+[0], 0, params))
                data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants[:-1]+[0], 350000, params, time)
                active = data[:,2]/params_constants[2]*100
                sims.append(active)
            except RuntimeWarning:
                print("Runtimewarning at idx: "+str(idx))
                # continue

            # else:

        # else:
        #     data = simulate_wt_experiment(model_fxns.nopos, initials, params_constants, 350000, params, time)

        if idx % int(len(top_params)*.1) == 0:
            print(str(int(idx/len(top_params)*100)) + "% complete.")
    return sims

# def simdata_to_list(model_fxns, top_params, params_constants, initials, time, param,
#                         ss = False):
#     sims = []
#     for idx, params in enumerate(top_params):
#         try:
#             if ss:
#                 ss_data = fsolve(model_fxns.m, initials, args=(0,params_constants, 0, params))
#                 data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, 350000, params, time, run_type=['ramp'])
#                 active = data[:,param]/params_constants[param]*100
#                 sims.append(active)
#             else:
#                 data = simulate_wt_experiment(model_fxns.m, initials, params_constants, 0, params, time, run_type=['man'])
#         except RuntimeWarning:
#             print("Runtimewarning at idx: "+str(idx))
#
#         if idx % int(len(top_params)*.1) == 0:
#             print(str(int(idx/len(top_params)*100)) + "% complete.")
#
#     return sims

def fit_m2c_data_to_list(model_fxns, top_params, params_constants, initials, time, param, dose, t100a=False, ptpD=False,
                        ss = False):
        sims = []
        for idx, params in enumerate(top_params):
            if ptpD:
                ss_data = run_ptpD_ss_M2c_ptp(model_fxns.m, initials, params_constants, params)
                check = params_constants[:-1] - ss_data[:-1]
                if (check < 0).any():
                    continue
                data = simulate_ptpD_experiment_M2c_ptp(model_fxns.m, ss_data, params_constants, dose, params, time)
            else:
                ss_data = run_ss_M2c_ptp(model_fxns.m, initials, params_constants, params)
                if t100a:
                    data = simulate_t100a_experiment_M2c_ptp(model_fxns.m, ss_data, params_constants, dose, params, time)
                else:
                    data = simulate_wt_experiment_M2c_ptp(model_fxns.m, ss_data, params_constants, dose, params, time ) #run_type=['ramp']
            if param == 3:
                active = [x/ss_data[3] for x in data[:,param]]
            else:
                active = data[:,param]/params_constants[param]*100
                if dose == 0:
                    false_ss = np.asarray([abs(active[0]-x) for x in active]) > 1
                    if false_ss.any():
                        continue
            sims.append(active)
        print('Dose: ' + str(dose) + ' complete.')
        return sims



def plt_param_cis_m2c_ptp(model_fxns, top_params, params_constants, initials,  doses, time, param,
                        exp_data=None, exp_time=None, ss=False, t100a=False, ptpD=False, ci= 95,
                        save_fig=''):
    plt.clf()
    font = {'family' : 'Arial',
    'weight' : 'normal',
    'size'   : 16}
    plt.rc('font', **font)

    if save_fig:
        fig, (ax1) = plt.subplots(1, 1, figsize=(2.25,2))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))

    plt.rc('ytick', labelsize=16)
    plt.rc('xtick', labelsize=16)

    palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#D9A673', 250000:'#B3804D', 350000: '#c4944d', 450000:'#794613', 550000:'#663300'}


    if param == 3:
        dt = 0.1
        steps = 1801
        time = np.linspace(0,dt*steps,steps)
    # elif doses == [0]:
    #     dt = 0.1
    #     steps = 3001
    #     time = np.linspace(0,dt*steps,steps)

    dashes= (2,2)
    for sig in doses:
        if ptpD:
            if t100a:
                sims = fit_m2c_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=True, ptpD=True, ss=ss)
            else:
                sims = fit_m2c_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=False, ptpD=True, ss=ss)
        else:
            if t100a:
                sims = fit_m2c_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=True, ptpD=False, ss=ss)
            else:
                sims = fit_m2c_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=False, ptpD=False, ss=ss)
        # if sig == 350000:
            # ax1 = sns.tsplot(sims, time,  ci = ci, color='#a97c50', dashes=dashes)
        # else:
        ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)

    mstyles = {0: ['full'],
                1: ['^', 'o'],
                2: ['D', 'o', '^', 'v', 's', 'D', 'o'],
                3: ['full'],
                4: ['full']}
    fstyles = {0: ['full'],
                1: ['full', 'bottom'],
                2: ['none', 'none'],
                3: ['full'],
                4: ['full']}
    if exp_data:
        if t100a:
            for sig, data in zip(doses, exp_data): #, mstyles.get(param)
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 550000:
                    mark = 's'
                else:
                    mark = 'o'
                ax1.plot(exp_time, data, 'o', marker=mark, markersize=6, fillstyle='none', color=palette.get(sig), mec='black', label = str(int(sig/1000)))
        else:
            for sig, data in zip(doses, exp_data):
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 550000:
                    mark = 's'
                else:
                    mark = 'o'
                ax1.plot(exp_time, data, 'o', marker=mark, markersize=6, fillstyle='full', color=palette.get(sig), mec='black', label = str(int(sig/1000)))#, fillstyle=fill, linestyle="None"), palette.get(sig), mec='black' (outside edge)





        # if plt_bad:
            # plt_thresh_behavior(model_fxns, top_params, plt_bad, params_constants, initials,  doses, time, param, ax1, ax2)

    if param == 3:
        ax1.set_xticks(np.arange(0, 181, step=60))
        ax1.set_yticks(np.arange(0, 5, step=1))
        ax1.set_ylim(1,4.25)
    elif ptpD:
        # ax1.set_ylim(50,105)
        ax1.set_yticks(np.arange(50, 101, step=10))
        ax1.set_xticks(np.arange(0, 61, step=15))
        # ax1.set_ylim(45,105)
        ax1.set_xlim(-2,62)
    # elif doses == [0]:
    #     ax1.set_ylim(-5,105)
    #     ax1.set_yticks(np.arange(0, 101, step=25))
    #     ax1.set_xticks(np.arange(0, 201, step=50))
    # else:
    #     ax1.set_yticks(np.arange(0, 101, step=25))
    #     ax1.set_xticks(np.arange(0, 61, step=15))
    #     ax1.set_ylim(-5,105)
    #     ax1.set_xlim(-2,62)

    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # ax1.legend(bbox_to_anchor=[1, 0.5], loc='best')

    if save_fig:
        plt.savefig(save_fig+".pdf", dpi=300,bbox_inches='tight')

    plt.show()

def fit_m4_data_to_list(model_fxns, top_params, params_constants, initials, time, param, dose, t100a=False, ptpD=False,
                        ss = False):
        sims = []
        for idx, params in enumerate(top_params):
            if ptpD:
                ss_data = run_ptpD_ss_M3_ptp(model_fxns.m, initials, params_constants, params)
                check = params_constants[:-1] - ss_data[:-1]
                if (check < 0).any():
                    continue
                data = simulate_ptpD_experiment_M3_ptp(model_fxns.m, ss_data, params_constants, dose, params, time)
            else:
                ss_data = run_ss(model_fxns.m, initials, params_constants, params)
                if t100a:
                    data = simulate_t100a_experiment_M3(model_fxns.m, ss_data, params_constants, dose, params, time)
                else:
                    data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, dose, params, time)
            if param == 3:
                active = [x/ss_data[3] for x in data[:,param]]
            else:
                active = data[:,param]/params_constants[param]*100
                if dose == 0:
                    false_ss = np.asarray([abs(active[0]-x) for x in active]) > 1
                    if false_ss.any():
                        continue
            sims.append(active)
        print('Dose: ' + str(dose) + ' complete.')
        return sims



def plt_param_cis_m4_ptp(model_fxns, top_params, params_constants, initials,  doses, time, param,
                        exp_data=None, exp_time=None, ss=False, t100a=False, ptpD=False, ci= 95,
                        save_fig=''):
    plt.clf()
    font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 16}
    plt.rc('font', **font)

    if save_fig:
        fig, (ax1) = plt.subplots(1, 1, figsize=(2.25,2))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(5,3))

    plt.rc('ytick', labelsize=16)
    plt.rc('xtick', labelsize=16)

    palette = {0:'#323232', 50000:'#D3D3D3', 150000:'#D9A673', 250000:'#B3804D', 350000: '#c4944d', 450000:'#794613', 550000:'#663300'}


    if param == 3:
        dt = 0.1
        steps = 1801
        time = np.linspace(0,dt*steps,steps)
    # elif doses == [0]:
    #     dt = 0.1
    #     steps = 3001
    #     time = np.linspace(0,dt*steps,steps)

    dashes= (2,2)

    for sig in doses:
        if ptpD:
            if t100a:
                sims = fit_m4_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=True, ptpD=True, ss=ss)
            else:
                sims = fit_m4_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=False, ptpD=True, ss=ss)
        else:
            if t100a:
                sims = fit_m4_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=True, ptpD=False, ss=ss)
            else:
                sims = fit_m4_data_to_list(model_fxns, top_params, params_constants, initials, time, param, sig, t100a=False, ptpD=False, ss=ss)
        ax1 = sns.tsplot(sims, time,  ci = ci, color=palette.get(sig), dashes=dashes)

    mstyles = {0: ['full'],
                1: ['^', 'o'],
                2: ['D', 'o', '^', 'v', 's', 'D', 'o'],
                3: ['full'],
                4: ['full']}
    fstyles = {0: ['full'],
                1: ['full', 'bottom'],
                2: ['none', 'none'],
                3: ['full'],
                4: ['full']}
    if exp_data:
        if t100a:
            for sig, data in zip(doses, exp_data): #, mstyles.get(param)
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 550000:
                    mark = 's'
                else:
                    mark == 'o'
                ax1.plot(exp_time, data, 'o', marker=mark, markersize=6, fillstyle='none', color=palette.get(sig), mec='black', label = str(int(sig/1000)))
        else:
            for sig, data in zip(doses, exp_data):
                if sig == 0:
                    mark = 'o'
                elif sig == 150000:
                    mark = '^'
                elif sig == 350000:
                    mark = 'v'
                elif sig == 550000:
                    mark = 's'
                else:
                    mark == 'o'
                ax1.plot(exp_time, data, 'o', marker=mark, markersize=6, fillstyle='full', color=palette.get(sig), mec='black', label = str(int(sig/1000)))#, fillstyle=fill, linestyle="None"), palette.get(sig), mec='black' (outside edge)





    # if plt_bad:
        # plt_thresh_behavior(model_fxns, top_params, plt_bad, params_constants, initials,  doses, time, param, ax1, ax2)

    if param == 3:
        ax1.set_xticks(np.arange(0, 181, step=60))
        ax1.set_yticks(np.arange(0, 5, step=1))
        ax1.set_ylim(1,4.25)
    elif ptpD:
        ax1.set_ylim(50,105)
        ax1.set_yticks(np.arange(50, 101, step=10))
        ax1.set_xticks(np.arange(0, 61, step=15))
        ax1.set_ylim(-5,105)
        ax1.set_xlim(-2,62)
    # elif doses == [0]:
    #     ax1.set_ylim(-5,105)
    #     ax1.set_yticks(np.arange(0, 101, step=25))
    #     ax1.set_xticks(np.arange(0, 201, step=50))
    else:
        ax1.set_yticks(np.arange(0, 101, step=25))
        ax1.set_xticks(np.arange(0, 61, step=15))
        ax1.set_ylim(-5,105)
        ax1.set_xlim(-2,62)

    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend(bbox_to_anchor=[1, 0.5], loc='best')

    if save_fig:
        plt.savefig(save_fig+".pdf", dpi=300,bbox_inches='tight')

    plt.show()


def plt_param_behaviors_m2c_ptp(model_fxns, top_params, plt_top, params_constants, initials,  doses, time, param,
                        mapk_wt_data=None, mapk_t100a_data=None, mapk_time=None, ptpD=False, ss=False, plt_bad=0,
                        save_fig=''):
    plt.clf()
    font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 20}
    plt.rc('font', **font)

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,4))

    # plot 1
#     title_text = 'Gen ' + str(gen) + ' best fits to WT'
    # title_text = 'Wild-type Simulations'
    # ax1.set_title(title_text, fontsize=20)
    # ax1.set_xlabel('Time (min)', fontsize=16)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    palette = palettes.get(param)
    # ax1.set_ylabel(x_labels.get(param), fontsize=24)
    ax1.set_yticks(np.arange(0, 101, step=25))
    ax1.set_xticks(np.arange(0, 61, step=15))

    if mapk_wt_data:
        # tdoses = [0, 150000, 350000, 550000]
        for sig, wt_data in zip(doses, mapk_wt_data):
            # ax1.plot(mapk_time, wt_data, 'o', markersize=10, color=palette.get(sig), label = str(int(sig/1000))+'mM KCl')
            ax1.plot(mapk_time, wt_data, 'o', markersize=10, color=palette.get(sig), label = str(int(sig/1000))+'mM KCl')

#     ax1.legend()

    # plot 2
#     title_text = 'Gen ' + str(gen) +  ' best fits to T100A + inhib'
    # title_text = 'Inhibited MAPK Simulations'#'Best fits to kinase dead mutant dose data'
    # ax2.set_title(title_text, fontsize=20)
    # ax2.set_xlabel('Time (min)', fontsize=16)
    # ax2.set_yticks(np.arange(0, 101, step=25))
    ax2.set_xticks(np.arange(0, 61, step=15))
    # ax2.set_ylabel(x_labels.get(param), fontsize=16)

    if mapk_t100a_data:
        for sig, t100a_data in zip(doses, mapk_t100a_data):
#             print(len(t100a_data))
#             if sig == 0:
#                 continue
#                 print(mapk_time_t100a_long)
#                 ax2.plot(mapk_time_t100a_long, t100a_data, '^', mew=2, markersize=10, color=palette.get(sig))
#             else:
            ax2.plot(mapk_time, t100a_data, '^', mew=2, markersize=10, color=palette.get(sig))


    if param == 3:
#         ax1.set_ylim(0,150)
        dt = 0.1
        steps = 2001
        time = np.linspace(0,dt*steps,steps)
        # ax1.set_yscale("log", nonposy='clip')
        # ax2.set_yscale("log", nonposy='clip')
    else:
        ax1.set_ylim(-5,105)
        ax2.set_ylim(-5,105)

#     if params_constants[-1] == 0:
#         ax1.set_ylim(50,105)
    if ptpD:

        for params in top_params[:plt_top]:
            # ptpD_total_protein = params_constants[:-2]+[550000*2, 0]
            # ptpD_inits = initials[:-1]+[0]

            ptpD_ss_inits = run_ptpD_ss_M2c_ptp(model_fxns.m, initials, params_constants, params)
            # ptpD_ss_inits = run_ss_ptps(model_fxns.m, initials, params_constants, params)

            # check = params_constants[:-1] - ptpD_ss_inits[:-1]
            # if (check < 0).any():
            #     continue
            # else:
                # mse_total = 0
            # ptp_doses = [0, 150000, 350000, 550000]
            ptp_doses = [0]

            plt.rc('xtick', labelsize=20)
            plt.rc('ytick', labelsize=20)
            # for dose, data in zip(doses, mapk_ptp_data):
            for sig in doses:
                odes = simulate_ptpD_experiment_M2c_ptp(model_fxns.m, ptpD_ss_inits, params_constants, sig, params, time)
                active = odes[:,param]/params_constants[param]*100
                false_ss = np.asarray([abs(active[0]-x) for x in active]) > 1
                if false_ss.any():
                    continue
                ax1.plot(time, active, color=pinks.get(sig), linewidth=3, alpha=.75)
                # data = model_fxns.t100a(model_fxns.m, ptpD_ss_inits, params_constants, sig, params, time)
                data = model_fxns.t100a(model_fxns.m, ptpD_ss_inits, params_constants, sig, params, time)
                active = data[:,param]/params_constants[param]*100
                ax2.plot(time, active, '--', color=palette.get(sig))
                # ax1.set_ylim(50,100)
                ax1.set_yticks(np.arange(50, 101, step=10))

    else:
        # ax1.plot(time, np.zeros(len(time)), color=palette.get(0))
        # ax2.plot(time, np.zeros(len(time)), '--', color=palette.get(0))

        for sig in doses:
            for params in top_params[:plt_top]:
                    if ss:
                        ss_data = run_ss_M2c_ptp(model_fxns.m, initials, params_constants, params)
                        data = simulate_wt_experiment_M2c_ptp(model_fxns.m, ss_data, params_constants, sig, params, time)
                    else:
                        data = simulate_wt_experiment_M2c_ptp(model_fxns.m, initials, params_constants, sig, params, time)
                    active = data[:,param]/params_constants[param]*100
                    ax1.plot(time, active, color=palette.get(sig))
                    if ss:
                        ss_data = run_ss_M2c_ptp(model_fxns.m, initials, params_constants, params)
                        data = model_fxns.t100a(model_fxns.m, ss_data, params_constants, sig, params, time)
                    else:
                        data = model_fxns.t100a(model_fxns.m, initials, params_constants, sig, params, time)
                    active = data[:,param]/params_constants[param]*100
                    ax2.plot(time, active, '--', color=palette.get(sig))

#     ax1.legend(bbox_to_anchor=[1, 0.5], loc='center left')
    ax1.grid(color='grey', linestyle='-', axis='y', linewidth=1)
    ax2.grid(color='grey', linestyle='-', axis='y', linewidth=1)
    if plt_bad:
        plt_thresh_behavior(model_fxns, top_params, plt_bad, params_constants, initials,  doses, time, param, ax1, ax2)
    if save_fig:
        plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/simulations/"+save_fig+".png",
        dpi=300,bbox_inches='tight')
    plt.show()

def plt_ramp_behaviors(model_fxns, top_params, plt_top, params_constants, initials, time, param,
                        ss = False, hog1_ramp_data=None, mapk_ramp_time=None,
                        save_fig=''):
    fig, (ax1) = plt.subplots(1, 1, figsize=(9,4))

    ax1.plot(mapk_ramp_time, hog1_ramp_data[0], 'o', markersize=10, color='Black')

    colors = sns.color_palette("bone")
    pal2 = sns.set_palette(colors)
    ax1.set_ylabel('% pp Hog1', fontsize=16)
    ax1.set_xlabel('Time (min)', fontsize=16)
    ax1.set_yticks(np.arange(0, 101, step=25))
    ax1.set_xticks(np.arange(0, 61, step=15))

    for params in top_params[:plt_top]:
#         for sig in doses:
            if ss:
                ss_data = run_ss(model_fxns.m, initials, params_constants, params)
                data = simulate_wt_experiment(model_fxns.m, ss_data, params_constants, 0, params, time, run_type=['ramp'])
            else:
                data = simulate_wt_experiment(model_fxns.m, initials, params_constants, 0, params, time, run_type=['ramp'])
            active = data[:,param]/params_constants[param]*100
            ax1.plot(time, active)
    if save_fig:
        plt.savefig("C:/Users/sksuzuki/Documents/Research/figures/simulations/"+save_fig+".png",
        dpi=300,bbox_inches='tight')
    plt.show()
