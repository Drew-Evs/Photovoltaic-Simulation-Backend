import numpy as np
from scipy.optimize import root, fsolve
import pvlib

#ignroe progress warnings
import warnings
warnings.filterwarnings('ignore')

#getting the guided initial guesses from the paper
def get_initial_guesses(specs):
    # Based on Section 4.3 Empirically Guided Guess Values https://asmedigitalcollection.asme.org/solarenergyengineering/article/134/2/021011/455696/An-Improved-Coefficient-Calculator-for-the
    tech = specs['tech'].lower().strip()
    n_s = specs['N_s']
    
    # Table 3: Technology-specific regressions for 'a' based on technology type
    a_map = {
        'mono-c-si': 0.027 * n_s - 0.0172,
        'multi-c-si': 0.026 * n_s + 0.0212,
        'thin film': 0.029 * n_s + 0.5264,
        'cdte': 0.012 * n_s + 1.356,
        'cigs': 0.018 * n_s + 0.327
    }
    a_guess = a_map.get(tech, 0.025 * n_s * 1.2)
    
    # Table 4: Empirical coefficients for Rs/Rsh
    c_map = {'mono-c-si': (0.32, 4.92),
             'multi-c-si': (0.34, 5.36), 
             'thin film': (0.59, 0.92),
             'cdte': (0.46, 1.11),
             'cigs': (0.55, 1.22)
    }

    cs, csh = c_map.get(tech, (0.5, 2.0))
    
    #initial guesses of a, il, io, rs, rsh, and adjust
    return [
        a_guess, specs['I_sc'], specs['I_sc'] * np.exp(-specs['V_oc']/a_guess),
        cs * (specs['V_oc'] - specs['V_mp']) / specs['I_mp'],
        csh * specs['V_oc'] / (specs['I_sc'] - specs['I_mp']),
        0.0    
    ]

#get the parameters based on changes in temperature and irradiance
def get_operational_params(x, T_c, S, specs):
    # Equations 3-6: Temperature and irradiacne
    #using the reference conditions currently to check against assumed changes
    a_ref, il_ref, io_ref, rs, rsh, adj = x
    t_ref, t_curr = 298.15, T_c + 273.15

    #boltzmanns constant in both ev and other
    k_ev = 8.617333e-5
    k = 1.380649e-23 #used for bandgap

    a_t = a_ref * (t_curr / t_ref)
    il_t = (S/1000) * (il_ref + specs['alpha_sc'] * (t_curr - t_ref))
    
    # Io temp dependency with Bandgap (Eq 5-6)
    eg_ref = 1.121
    eg_t = eg_ref * (1 - 0.0002677 * (t_curr - t_ref))
    io_t = io_ref * (t_curr/t_ref)**3 * np.exp((eg_ref/(k_ev*t_ref)) - (eg_t/(k_ev*t_curr)))

    rsh_t = rsh * (1000/S)
    
    return a_t, il_t, io_t, rs, rsh_t

#to find the gamma value of the 6 parameter model ensure it matches that given by the datasheet
def calculate_gamma_model(x, S, specs):
    #testing all powers between increments 
    temps = np.arange(-10, 51, 3)
    p_maxes = []
    p_mp_src = specs['V_mp'] * specs['I_mp']
    
    for t in temps:
        a_t, il_t, io_t, rs, rsh = get_operational_params(x, t, S, specs)
        def mpp_func(z):
            v, i = z
            g_d = (io_t/a_t) * np.exp((v + i*rs)/a_t) + (1/rsh)
            # Eq 24 & Eq 23 residuals
            return [
                il_t - io_t*(np.exp((v+i*rs)/a_t)-1) - (v+i*rs)/rsh - i,
                i - v * (g_d / (1 + rs*g_d))
            ]
        vm, im = fsolve(mpp_func, [specs['V_mp'], specs['I_mp']])
        p_maxes.append(vm * im)
        
    # Average slope (Eq 10)
    slopes = [(p_maxes[i+1]-p_maxes[i])/(p_mp_src*(temps[i+1]-temps[i])) for i in range(len(p_maxes)-1)]
    return np.mean(slopes)

def cec_6_residual(x, S, specs):
    a, il, io, rs, rsh, adj = x
    # Standard residuals (Eq 28)
    f = np.zeros(6)
    f[0] = il - io*(np.exp(specs['I_sc']*rs/a)-1) - (specs['I_sc']*rs/rsh) - specs['I_sc']
    f[1] = il - io*(np.exp(specs['V_oc']/a)-1) - (specs['V_oc']/rsh)
    f[2] = il - io*(np.exp((specs['V_mp']+specs['I_mp']*rs)/a)-1) - (specs['V_mp']+specs['I_mp']*rs)/rsh - specs['I_mp']
    
    # Derivative constraint (Eq 14)
    g_d = (io/a) * np.exp((specs['V_mp']+specs['I_mp']*rs)/a) + (1/rsh)
    f[3] = specs['I_mp'] - specs['V_mp'] * (g_d / (1 + rs*g_d))
    
    # Temp corrections (Eq 22 & 26)
    # Voc at T+5C
    voc_t = specs['beta_oc']*(1 + adj/100)*5.0 + specs['V_oc']
    at, ilt, iot, _, _ = get_operational_params(x, 25 + 5.0, S, specs)
    f[4] = ilt - iot*(np.exp(voc_t/at)-1) - (voc_t/rsh)
    
    # Power slope (gamma)
    f[5] = specs['gamma'] - calculate_gamma_model(x, S, specs)
    return f

def trial(guess_curr, pvl_array, t_c, S, specs):
    sol = root(cec_6_residual, guess_curr, args=(1000, specs,))

    #testing solution passes the heuristic tests
    if heuristic_test(sol, specs) == True:

        a_ref, il_ref, io_ref, rs, rsh, adjust = sol.x

        #then find at conditions
        our_params = get_operational_params((a_ref, il_ref, io_ref, rs, rsh, adjust), t_c, S, specs)
        reordered_params = np.array([our_params[1], our_params[2], our_params[3], our_params[4], our_params[0]])
    
        #calculate rms to see how far away
        residual = (reordered_params - pvl_array) / pvl_array
        rms_error = np.sqrt(np.mean(residual**2))

        return rms_error
    
    return False

#similar to heuristic solver except returns parameters for a module
def param_solver(specs, t_c, S):
    #create an initial guess
    x0 = get_initial_guesses(specs)

    #try raw guess first 
    result = param_calc(x0, t_c, S, specs)

    if result is not False:
        return result

    #removed first 2 heuristics due to lack of improvement

    #irrespective of technology if still not converging - increase Isc by factor of 1%
    initial_specs = specs.copy()
    for i in range(6):
        multiplier = float(f'1.0{i}')
        specs['I_sc'] = initial_specs['I_sc'] * multiplier

        x_new = get_initial_guesses(specs)
        result = param_calc(x_new, t_c, S, specs)
        if result is not False:
            return result

    return False

#similar to trial but returns reorderd params instead of residuals
def param_calc(guess_curr, t_c, S, specs):
    sol = root(cec_6_residual, guess_curr, args=(1000, specs,))

    #testing solution passes the heuristic tests
    if heuristic_test(sol, specs, t_c) == True:

        a_ref, il_ref, io_ref, rs, rsh, adjust = sol.x

        #then find at conditions
        our_params = get_operational_params((a_ref, il_ref, io_ref, rs, rsh, adjust), t_c, S, specs)

        return our_params
    
    return False

def heuristic_test(sol, specs, t_c):  
    #using i-v law to calc voltage
    def iv_equation(I, V):
        exponent = (V + I * rs)/(a)
        exponent_term = isat * (np.exp(np.clip(exponent, -50, 50)) -1)
        rsh_term = (V + I * rs)/rsh
        return (iph - exponent_term - rsh_term - I)
  
    bounds = [
        (0.05, 15),
        (0.5, 15),
        (1e-16, 1e-7),
        (0.001, 75),
        (1, 10000),
        (-100, 100)
    ]

    a, iph, isat, rs, rsh, adjust = sol.x

    #checking within bounds
    params = np.array([a, iph, isat, rs, rsh, adjust])

    # tolerance for floating point comparison
    tol = 1e-6

    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    inside_bounds = np.all((params >= lower_bounds) & (params <= upper_bounds))

    if not inside_bounds:
        return False

    #generating the curve to see if points match
    voltages = np.linspace(0.015*specs['V_oc'], 0.98*specs['V_oc'], 200)
    currents = []
    powers = []

    #iterate guess over time
    curr_guess = specs['I_mp']
    for V in voltages:
        sol = fsolve(iv_equation, curr_guess, args=(V,))[0]

        currents.append(sol)
        powers.append(sol*V)

        curr_guess = sol

    #testing max power matches that of the module
    idx = np.argmax(powers)
    pmp = powers[idx]
    imp = currents[idx]
    model_pmp = specs['V_mp'] * specs['I_mp']
    
    if abs(pmp - model_pmp) > (0.015 * model_pmp):
        return False
    
    #testing derivative of curve is alwasy <=  0
    dIdV = np.gradient(currents, voltages)

    if np.any(dIdV > 1e-6):
        return False
    
    return True

def getting_parameters(t_c, S, module_name):
    cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
    module = cec_modules[module_name]

    specs = {
        'tech': module['Technology'],
        'N_s': module['N_s'],
        'I_sc': module['I_sc_ref'],
        'V_oc': module['V_oc_ref'],
        'I_mp': module['I_mp_ref'],
        'V_mp': module['V_mp_ref'],
        'alpha_sc': module['alpha_sc'],
        'beta_oc': module['beta_oc'],
        'gamma': module['gamma_r']/100
    }
    
    params = param_solver(specs, t_c, S)

    out_params = np.array([params[1], params[2], params[3], params[4], params[0]])
    return out_params

    
def getting_parameters_specs(t_c, S, specs):
    params = param_solver(specs, t_c, S)

    out_params = np.array([params[1], params[2], params[3], params[4], params[0]])
    return out_params

# temp_cell = 45
# irradiance = 500
# cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')

# module_list = []
# count = 0
# correct_count = 0
# accurate_count = 0

# #record rms against modules
# recorded_rms = []
# recorded_modules = []

# #rms tolerance 
# tol = 1

# min_res = np.inf
# min_name = ''

# for module_name in cec_modules.columns[::50]:
#     module = cec_modules[module_name] 

#     specs = {
#         'tech': module['Technology'],
#         'N_s': module['N_s'],
#         'I_sc': module['I_sc_ref'],
#         'V_oc': module['V_oc_ref'],
#         'I_mp': module['I_mp_ref'],
#         'V_mp': module['V_mp_ref'],
#         'alpha_sc': module['alpha_sc'],
#         'beta_oc': module['beta_oc'],
#         'gamma': module['gamma_r']/100
#     }

#     output = heuristic_solver(specs, temp_cell, irradiance, module_name)

#     #need to ensure not 0
#     if output is not False:
#         correct_count += 1
#         if output <= tol:
#             accurate_count += 1
#             recorded_modules.append(module_name)
#             recorded_rms.append(output)

#             if output < min_res:
#                 min_res = output
#                 min_name = module_name

#     count += 1

# import matplotlib.pyplot as plt
# import pandas as pd

# results_df = pd.DataFrame({
#     "Module": recorded_modules,
#     "RMS_Error": recorded_rms
# })

# results_df = results_df.sort_values("RMS_Error")

# num_bins = 30

# plt.figure(figsize=(12, 8))  # histogram width can be static or dynamic
# plt.hist(results_df["RMS_Error"], bins=num_bins, edgecolor='black', alpha=0.7)

# plt.xlabel("RMS Relative Parameter Error")
# plt.ylabel("Number of Modules")
# plt.title("Distribution of Heuristic Solver RMS Errors")
# plt.grid(axis='y', alpha=0.75)
# plt.tight_layout()
# plt.savefig("graphs/residual_modules_histogram.png", dpi=300)

# print(f'Percent convergence: {(correct_count/count)*100:.2f}%')
# print(f'Converged with a low RMSE to actual conditions: {(accurate_count/count)*100:.2f}%')
# print(f'Most accurate module is {min_name} at {min_res}')