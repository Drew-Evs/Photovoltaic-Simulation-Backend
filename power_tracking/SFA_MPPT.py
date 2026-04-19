import numpy as np
import pvlib
from power_tracking.refactored_whole_module import Module
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import concurrent.futures

'''as a class to make MPPT tracking simpler'''
class DPSO_MPPT:
    def __init__(self, num_fireflies, module, RL_min, RL_max, RPV_min=3.06, RPV_max=25.53, nbb=1):
        #the module to track the max power of
        self.module = module

        #the number of fireflies to track
        self.num_fireflies = num_fireflies
        #the efficiency of the converter (assume ~0.95) 
        self.nbb = nbb
        #the min and max load
        self.RL_min = RL_min
        self.RL_max = RL_max
        #min and max reflective impedance - calculated in previous section
        self.RPV_min = RPV_min
        self.RPV_max = RPV_max

        #use duty cycel of between 0 and 1
        self.d_min = 0.05
        self.d_max = 0.95

        self.v_out = module.voc

        #initiate duty cycles for each particle and their velocity
        self.fireflies = np.linspace(self.d_min, self.d_max, self.num_fireflies)
        self.velocities = np.zeros(self.num_fireflies)

        #only track global best (most attractive firefly)
        self.gbest_position = self.fireflies[0]
        self.gbest_power = 0

        #the beta coefficient (initiate at 0.3 and max of 3)
        self.beta = 0.3
        self.beta_max = 1.5
        

    #use the formulas discussed to calculate min and max
    def calculate_d_min(self):
        numer = np.sqrt(self.nbb * self.RL_min)
        denom = np.sqrt(self.RPV_max) + np.sqrt(self.nbb * self.RL_min)
        return numer/denom
    
    def calculate_d_max(self):
        numer = np.sqrt(self.nbb * self.RL_max)
        denom = np.sqrt(self.RPV_min) + np.sqrt(self.nbb * self.RL_max)
        return numer/denom

    #used to convert from duty cycle to equivalent voltage
    def get_voltage(self, D):
        return self.v_out*(1-D)
    
    #find the best duty cycle position
    def global_optimisation(self):
        #reset for new tracking
        self.fireflies = np.linspace(self.d_min, self.d_max, self.num_fireflies)
        self.gbest_position = self.fireflies[0]
        self.gbest_power = 0

        #initial currents and voltages
        current_guesses = np.linspace(0, self.module.isc, self.num_fireflies)
        voltages = [self.get_voltage(D) for D in self.fireflies]

        #bounds for the solver
        low_bound = np.array([-10.0]*self.module.Ns + [-np.inf]*(self.module.Ns + 2*self.module.d) + [0])
        low_bound[self.module.Ns:2*self.module.Ns] = 0
        high_bound = np.array([np.inf]*(2*self.module.Ns + 2*self.module.d + 1))
        high_bound[0:self.module.Ns] = [self.module.voc_per_cell]*self.module.Ns
        bounds = (low_bound, high_bound)

        #initial guesses for the solver
        guesses = []
        for i in range(self.num_fireflies):
            temp_guess = np.concatenate([
                [voltages[i]/self.module.Ns]*self.module.Ns, [current_guesses[(self.num_fireflies-1)-i]]*self.module.Ns,
                [0.0]*self.module.d, [0.0]*self.module.d, [current_guesses[(self.num_fireflies-1)-i]]
            ])

            guesses.append(temp_guess)

        #using a tolerance to end loop
        max_iter = 100
        iter = 0
        tol = 1e-2

        #skip if power not meaningfully changing
        power_tol = 1e-6
        #if voltage barely changes then want to skip rerunning solver
        voltage_threshold = 0.05

        #store powers/force initial run
        powers = np.zeros(self.num_fireflies)
        previous_voltages = np.zeros(self.num_fireflies) - 999
        stall_counter = 0

        #history for graph
        history = []

        #optimise series of steps
        while iter < max_iter:
            #update voltages by duty cycle
            voltages = [self.get_voltage(D) for D in self.fireflies]
            voltages = np.clip(voltages, 0, self.v_out).tolist()

            #update powers and guesses by previous result
            #add to queue for parallelisation
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_fireflies) as executor:
                future_to_idx = {}
                for i, V in enumerate(voltages):
                    #only update if voltage changes significantly
                    if abs(V-previous_voltages[i]) > voltage_threshold:
                        #submit to solver as a background thread 
                        future = executor.submit(self.module.PSO_method, V, guesses[i], bounds)
                        future_to_idx[future] = i

                #collect results as they finish
                for future in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[future]
                    power, temp_guess = future.result()
                    powers[i] = power
                    guesses[i] = temp_guess

            previous_voltages = np.copy(voltages)

            history.append({
                'voltages': list(voltages),
                'powers': powers.copy()
            })

            #store to check for power stalling before optimising
            old_gbest_power = self.gbest_power
            self.optimise_step(powers, iter)

            #increase stall if iteration not improved
            if (self.gbest_power - old_gbest_power) < power_tol:
                stall_counter += 1
                #power needs to stay the same for 6 loops
                if stall_counter >= 6:
                    print(f'Converged on power after {iter} iterations.')
                    break
            else:
                stall_counter = 0

            iter += 1
        
        #print(f'Best voltage is {self.get_voltage(self.gbest_position)} and best power is {self.gbest_power}')
            if iter == max_iter:
                print(f'Didnt converge within max iterations')

        return self.get_voltage(self.gbest_position), self.gbest_power, history

    #each individual step of the optimisation
    def optimise_step(self, current_powers, current_iter):
        #global evaluation - find brightest firefly
        best_idx = np.argmax(current_powers)
        if current_powers[best_idx] > self.gbest_power:
            self.gbest_power = current_powers[best_idx]
            self.gbest_position = self.fireflies[best_idx]

        #update movement based on brightest
        movement = self.beta * (self.gbest_position - self.fireflies)
        self.fireflies += movement

        #nudge the best by a tiny amount to keep explorint
        nudge_amount = 0.005 
        if current_iter % 2 == 0:
            self.fireflies[best_idx] += nudge_amount
        else:
            self.fireflies[best_idx] -= nudge_amount

        #increase beta by 0.25 each time
        self.beta += 0.25
        self.beta = np.clip(self.beta, 0, self.beta_max)

        #vectorised clipping
        self.fireflies = np.clip(self.fireflies, self.d_min, self.d_max)

    #iterate through condition arrays
    def set_module_conditions(self, temp_array=[], irr_array=[]):
        self.module.set_cell_conditions(temp_array, irr_array)

#for profiling
import cProfile
import pstats
import io

#testing accuracy and performance against the full loop
def test_accuracy(module):
    pr = cProfile.Profile()
    pr.enable()

    module_tracker = DPSO_MPPT(5, module, 0, module.voc)
    shaded_cells = np.array([[6,11], [43,47]])
    shade_level = 250

    irr_array = np.full(module.Ns, 1000)
    for start, end in shaded_cells:
        irr_array[start:end+1] = shade_level 

    module_tracker.set_module_conditions(irr_array=irr_array)
    module.set_cell_conditions(irr_array=irr_array) 

    print("-"*50)
    print("Starting tracker")
    tracker_vmp, tracker_pmp, history = module_tracker.global_optimisation()
    print("Starting full curve")
    _, curve_voltages, curve_powers = module.calculate_iv()
    module_pmp = curve_powers[np.argmax(curve_powers)]

    print(f'Tracker pmp is {tracker_pmp} and actual is {module_pmp}')
    print(f'Difference between tracker and actual is {np.abs(tracker_pmp - module_pmp)}')

    plt.figure(figsize=(10, 6))
    
    # 1. Plot the continuous P-V curve as the baseline
    # (Optional: If the black line looks jagged like the first graph, remember 
    #  to sort curve_voltages and curve_powers before plotting)
    plt.plot(curve_voltages, curve_powers, label='Actual P-V Curve', color='black', linewidth=2, zorder=1)

    # 2. Scatter the particle history - Colored by PARTICLE instead of ITERATION
    num_particles = len(history[0]['voltages'])
    # Using 'tab10' which gives distinct, bright colors for different categories/particles
    colors = cm.tab10(np.linspace(0, 1, num_particles))

    # Loop through each individual particle index
    for p_idx in range(num_particles):
        # Extract the voltage and power for this specific particle across ALL iterations
        p_voltages = [step['voltages'][p_idx] for step in history]
        p_powers = [step['powers'][p_idx] for step in history]
        
        # Plot all steps for this particle. 
        # By omitting 'label', it will not appear in the legend.
        plt.scatter(p_voltages, p_powers, color=colors[p_idx], s=50, alpha=0.7, zorder=2)

    # 3. Mark the final Global Best found by the algorithm
    plt.scatter(tracker_vmp, tracker_pmp, color='red', marker='*', s=200, label='SFA Final MPPT', zorder=3)

    plt.title('SFA Particles Converging on P-V Curve (Partial Shading)')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the file
    filename = 'sfa_convergence.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Convergence graph successfully saved as '{filename}'")
    plt.close()

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

    module = Module(datasheet_conditions, 'Prism_Solar_Technologies_Bi48_267BSTC')
    
    #optimise_parameters(module)
    test_accuracy(module)