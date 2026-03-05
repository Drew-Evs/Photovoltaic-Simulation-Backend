from refactored_whole_module import Module
import numpy as np
import pvlib

#process - pick 4 random points and then update them until MPP found
def track_mpp_pso(module):
    #v_out using the voc
    v_out = 35

    '''need more references for this part'''
    #calculate the min/max duty cycles
    #use fixed min/max
    d_min = 0.1
    d_max = 0.9

    #initial guesses for the 4
    duty_cycles = np.linspace(d_min, d_max, 4)
    current_guesses = np.linspace(0, module.isc, 4)
    guesses = []

    #generate and clip intiital voltages
    voltages = np.linspace(0, module.voc, 4)
    voltages = np.clip(voltages, 0.01, module.voc)

    #bounds for the solver
    low_bound = np.array([-10.0]*module.Ns + [-np.inf]*(module.Ns + 2*module.d) + [0])
    low_bound[module.Ns:2*module.Ns] = 0
    high_bound = np.array([np.inf]*(2*module.Ns + 2*module.d + 1))
    high_bound[0:module.Ns] = [module.voc_per_cell]*module.Ns
    bounds = (low_bound, high_bound)

    for i in range(4):
        #voltages and currents are opposite
        temp_guess = np.concatenate([
            [voltages[i]/module.Ns]*module.Ns, [current_guesses[3-i]]*module.Ns,
            [0.0]*module.d, [0.0]*module.d, [current_guesses[3-i]]
        ])

        guesses.append(temp_guess)

    #have a tolerance/max number of iterations
    tol = 1e-1
    max_iter = 100
    iter = 0
    v_max = 0.10

    #weighting for PSO
    w = 0.4 #inertia
    c1 = 1.0 #personal weight
    c2 = 1.0 #social weight 

    #start with initial velocity of 0
    velocities = np.zeros(4)
    #have local (personal best) powers/voltages
    p_best_cycles = np.copy(duty_cycles)
    p_best_powers = np.zeros(4)

    #and global best
    g_best_cycle = 0.0
    g_best_power = 0.0

    print("Running tracking")
    while iter < max_iter:
        iter += 1
        #get the voltages from the duty cycles
        voltages = [v_out*(1-D) for D in duty_cycles]

        #generate new powers - check for best
        temp_powers = np.zeros(4)
        for i in range(4):
            temp_powers[i], guesses[i] = module.PSO_method(voltages[i], guesses[i], bounds)

        max_idx = np.argmax(temp_powers)
        max_result = temp_powers[max_idx]

        if np.abs(max_result-g_best_power) < tol:
            print(f"Converged to MPP at duty cycle: {g_best_cycle:.2f}")
            break

        #update personal bests
        for i, (t_power, p_power) in enumerate(zip(temp_powers, p_best_powers)):
            if t_power > p_power:
                p_best_powers[i] = temp_powers[i]
                p_best_cycles[i] = duty_cycles[i]

        #update global best
        if max_result > g_best_power:
            g_best_power = max_result
            g_best_cycle = duty_cycles[max_idx]

        #update velocities using cognitive and social (congitive impacted by personal)
        #social impacted by global
        for i in range(4):
            cognitive = c1 * (p_best_cycles[i] - duty_cycles[i])
            social = c2 * (g_best_cycle - duty_cycles[i])
    
            velocities[i] = w * velocities[i] + cognitive + social
            
            #max jump of 10%
            #velocities[i] = np.clip(velocities[i], -v_max, v_max)
            duty_cycles[i] = duty_cycles[i] + velocities[i]

            #and clamp to 0.1 and 0.9
            #duty_cycles[i] = np.clip(duty_cycles[i], d_min, d_max)

    return v_out*(1-g_best_cycle), g_best_power

'''want to calculate impedance at minimum and maximum i.e. at best conditions and worst conditions'''
'''formula is V/I - so need to calculate pmp at both conditions - can save and store - only needs to be run once'''
'''when run: Max PV = 3.0559068023081455 and Min PV = 25.532007401360936'''
def min_max_rpv(datasheet_conditions, module_name):
    #get maximum first - 1000W/m2 and 25C (STC)
    module=Module(datasheet_conditions, module_name)
    currents, voltages, powers = module.calculate_iv()
    max_idx = np.argmax(powers)
    pv_max = voltages[max_idx]/currents[max_idx]

    #then minimum - 100W/m2 and 65C
    for i in range(module.Ns):
        module.cell_list[i].set_temp(65)
        module.cell_list[i].shade(100)

    module.update_cell_arrays()
    currents, voltages, powers = module.calculate_iv()
    max_idx = np.argmax(powers)
    pv_min = voltages[max_idx]/currents[max_idx]

    print(f'Max PV = {pv_max} and Min PV = {pv_min}')

#for profiling
import cProfile
import pstats
import io

def run_profile():
    pr = cProfile.Profile()

    pr.enable()

    cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
    module = cec_modules['Prism_Solar_Technologies_Bi48_267BSTC']
    module_name = 'Prism_Solar_Technologies_Bi48_267BSTC'
    datasheet_conditions = (
        module['I_sc_ref'], 
        module['V_mp_ref'], 
        module['V_oc_ref'], 
        module['I_mp_ref'],
        module['N_s']
    )

    #create the module
    module = Module(datasheet_conditions, module_name)

    shaded_cells = np.array([[6,11], [43,47]])
    shade_level = 250

    for start, end in shaded_cells:
        for i in range(start, end + 1):
            module.cell_list[i].shade(shade_level)

    #update the arrays used in the least square
    module.update_cell_arrays()

    shaded_cells = np.array([[6,11], [43,47]])
    shade_level = 250
    voltage, power = track_mpp_pso(module)

    print(f"Max power point is {power}W")

    pr.disable()

    #sort values
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)

    print(s.getvalue())


if __name__ == "__main__":
    cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
    module = cec_modules['Prism_Solar_Technologies_Bi48_267BSTC']
    module_name = 'Prism_Solar_Technologies_Bi48_267BSTC'
    datasheet_conditions = (
        module['I_sc_ref'], 
        module['V_mp_ref'], 
        module['V_oc_ref'], 
        module['I_mp_ref'],
        module['N_s']
    )

    min_max_rpv(datasheet_conditions, module_name)
    # run_profile()

    # best_voltage, best_power = track_mpp_pso(module)

    # currents, voltages, powers = module.calculate_iv("MPP Test")

    # #calculating the whole iv curve against MPP method
    # max_idx = np.argmax(powers)
    # iv_voltage = voltages[max_idx]
    # iv_power = powers[max_idx]

    # print(f'IV curve best power = {iv_power} and MPP method = {best_power}')
    # print(f'IV curve best voltage = {iv_voltage} and MPP method = {best_voltage}')



