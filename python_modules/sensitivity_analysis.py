import numpy as np
import model_supp


def scan_param(param_vec, param_to_change, scan_range):
    scores = []
    for x in scan_range:
        _param = step_size(param_vec,param_to_change,x)
        _score = model_supp.calc_sim_score(model_fxns, exp_data, exp_time, params_constants, initials, _param, ptpD=False, full=True, convert = (False,))
        scores.append(_score)
    return np.array(scores)

def calc_norm_sensitivity(param_vec, param_to_change, original_mse, step=0.001):
    #step_size as a percentage of the param value
#     for i in [+, -]:
    param_1 = param_vec.copy()
    param_1[param_to_change] = param_1[param_to_change] + param_1[param_to_change]*step #0.001
    mse_1 = model_supp.calc_sim_score(model_fxns, exp_data, exp_time, params_constants, initials, param_1, ptpD=False, full=True, convert = (False,))
    mse_1 = sum(mse_1[:18])

    param_2 = param_vec.copy()
    param_2[param_to_change] = param_2[param_to_change]-param_2[param_to_change]*step #0.001
    mse_2 = model_supp.calc_sim_score(model_fxns, exp_data, exp_time, params_constants, initials, param_2, ptpD=False, full=True, convert = (False,))
    mse_2 = sum(mse_2[:18])
    #     print(original_mse)

#     sensitivity_norm = ((mse_1-mse_2)/original_mse)/(2*step/param_vec[param_to_change])
#     sensitivity_norm = (np.abs(mse_1-mse_2)/original_mse)/(2*step)

#     sensitivity_norm = ((mse_1-original_mse)+(mse_2-original_mse)/original_mse)/(2*step/param_vec[param_to_change])
#     sensitivity_norm = ((mse_1-original_mse)+(mse_2-original_mse))/(2*step)
#     sensitivity_norm = ((mse_1+mse_2)/2/original_mse)
#     sensitivity_norm = ((mse_1-mse_2)/(2*step))*(param_vec[param_to_change]/original_mse) # correct one
    sensitivity_norm = ((mse_1-mse_2)/(2*step))*(param_vec[param_to_change]/original_mse) # correct one

    return sensitivity_norm

def sa(param_set,pred=False):
    sa_values = []
    for param_s in param_set:
#         original_mses = []
        data = []
        original_mse = sum(model_supp.calc_sim_score(model_fxns, exp_data, exp_time, params_constants, initials, param_s, ptpD=False, full=True, convert = (False,))[:18])
#         print(mse)
        for param in range(len(param_s)):
#             if pred:
#                 _mse = mse[18]
#             else:
#                 _mse = sum(mse[:18])#+mse[19]
#             original_mses.append(_mse)

            sensitivity_norm = calc_norm_sensitivity(param_s, param, original_mse, step=0.001)
#             print(sensitivity_norm)
#             if pred:
#                 t_mse = [x[18] for x in mses]
#             else:
#                 t_mse = [sum(x[:18]) for x in mses] #+x[19]

            data.append(sensitivity_norm)

#         sum_mse = np.sum(np.array(data), axis = 1)/np.array(original_mses)
        sa_values.append(data)
    return np.array(sa_values)
