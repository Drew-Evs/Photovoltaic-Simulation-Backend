import numpy as np
import pvlib
from refactored_whole_module import Module
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import concurrent.futures

'''as a class to make MPPT tracking simpler'''
class DPSO_MPPT:
    def __init__(self, num_particles, module, RL_min, RL_max, RPV_min=3.06, RPV_max=25.53, nbb=1):
        #the module to track the max power of
        self.module = module

        #the number of particles to track
        self.num_particles = num_particles
        #the efficiency of the converter (assume ~0.95) 
        self.nbb = nbb
        #the min and max load
        self.RL_min = RL_min
        self.RL_max = RL_max
        #min and max reflective impedance - calculated in previous section
        self.RPV_min = RPV_min
        self.RPV_max = RPV_max

        #calculate the duty cycles tracked between
        self.d_min = self.calculate_d_min()
        self.d_max = self.calculate_d_max()

        self.v_out = module.voc

        #initiate duty cycles for each particle and their velocity
        self.particles = np.linspace(self.d_min, self.d_max, self.num_particles)
        self.velocities = np.zeros(self.num_particles)

        #tracck population personal bests and global best
        self.pbest_positions = np.copy(self.particles)
        self.pbest_powers =  np.zeros(self.num_particles)
        self.gbest_position = self.particles[0]
        self.gbest_power = 0

        #using weights (inertia cognitive and social)
        # self.c1, self.c2, self.w = 0.4, 0.1, 0.4
        #just an inertia weight
        self.w = 0.3
        self.v_max = 0.05 

        #whether tracking globally or locally
        self.state = 'Global'
        self.po_step_size = 0.01

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
        self.particles = np.linspace(self.d_min, self.d_max, self.num_particles)
        self.velocities = np.zeros(self.num_particles)
        self.pbest_positions = np.copy(self.particles)
        self.pbest_powers =  np.zeros(self.num_particles)
        self.gbest_position = self.particles[0]
        self.gbest_power = 0

        #initial currents and voltages
        current_guesses = np.linspace(0, self.module.isc, self.num_particles)
        voltages = [self.get_voltage(D) for D in self.particles]

        #bounds for the solver
        low_bound = np.array([-10.0]*self.module.Ns + [-np.inf]*(self.module.Ns + 2*self.module.d) + [0])
        low_bound[self.module.Ns:2*self.module.Ns] = 0
        high_bound = np.array([np.inf]*(2*self.module.Ns + 2*self.module.d + 1))
        high_bound[0:self.module.Ns] = [self.module.voc_per_cell]*self.module.Ns
        bounds = (low_bound, high_bound)

        #initial guesses for the solver
        guesses = []
        for i in range(self.num_particles):
            temp_guess = np.concatenate([
                [voltages[i]/self.module.Ns]*self.module.Ns, [current_guesses[(self.num_particles-1)-i]]*self.module.Ns,
                [0.0]*self.module.d, [0.0]*self.module.d, [current_guesses[(self.num_particles-1)-i]]
            ])

            guesses.append(temp_guess)

        #using a tolerance to end loop
        max_iter = 100
        iter = 0
        tol = 1e-4

        #skip if power not meaningfully changing
        power_tol = 1e-6
        #if voltage barely changes then want to skip rerunning solver
        voltage_threshold = 0.05

        #store powers/force initial run
        powers = np.zeros(self.num_particles)
        previous_voltages = np.zeros(self.num_particles) - 999
        stall_counter = 0

        #history for graph
        history = []

        #optimise series of steps
        while iter < max_iter:
            #update voltages by duty cycle
            voltages = [self.get_voltage(D) for D in self.particles]

            #update powers and guesses by previous result
            #add to queue for parallelisation
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_particles) as executor:
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
            self.optimise_step(powers)

            #early stop check
            #if variance between all particle positions are basically 0
            if np.std(self.particles) < tol:
                print(f'Converged after {iter} iterations')
                break
            
            #increase stall if iteration not improved
            if (self.gbest_power - old_gbest_power) < power_tol:
                stall_counter += 1
                #power needs to stay the same for 3 loops
                if stall_counter >= 4:
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
    def optimise_step(self, current_powers):
        #global dpso 
        #changed to vector and masking to improve speed
        mask = current_powers > self.pbest_powers
        self.pbest_powers[mask] = current_powers[mask]
        self.pbest_positions[mask] = self.particles[mask]

        #global evaluation
        best_idx = np.argmax(current_powers)
        if current_powers[best_idx] > self.gbest_power:
            self.gbest_power = current_powers[best_idx]
            self.gbest_position = self.particles[best_idx]

        #updated velocity calculation - and clipping
        self.velocities = (self.w*self.velocities) + (self.gbest_position + self.pbest_positions - 2*self.particles)
        self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
        self.particles += self.velocities

        #vectorised clipping
        self.particles = np.clip(self.particles, self.d_min, self.d_max)

    #iterate through condition arrays
    def set_module_conditions(self, temp_array=[], irr_array=[]):
        self.module.set_cell_conditions(temp_array, irr_array)

    #hybride tracker - tracks either globally if theres a large change in power or locally if minimal
    def track_mpp(self):
        #runs the global optimisation method if in DPSO state
        if self.state == 'Global':
            print("Large Change Detected - Running Global Tracking")
            best_v, best_p, history = self.global_optimisation()

            #lock results into last p&o vars
            self.last_po_pos = self.gbest_position
            self.last_po_power = best_p

            #switch to P&O for future
            self.state = 'Local'

        elif self.state == 'Local':
            #take a small step 
            new_position = self.last_po_position + (self.po_direction*self.po_step_size)
            new_position = np.clip(new_position, self.d_min, self.d_max)

            #evaluate at the new position
            new_power = self.evaluate_single_position(new_position)


import random

#for profiling
import cProfile
import pstats
import io

#testing accuracy and performance against the full loop
def test_accuracy(module):
    pr = cProfile.Profile()
    pr.enable()

    module_tracker = DPSO_MPPT(module.d-1, module, 0, module.voc)
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
    plt.scatter(tracker_vmp, tracker_pmp, color='red', marker='*', s=200, label='DPSO Final MPPT', zorder=3)

    plt.title('DPSO Particles Converging on P-V Curve (Partial Shading)')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the file
    filename = 'dpso_convergence.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Convergence graph successfully saved as '{filename}'")
    plt.close()

    pr.disable()

    #sort values
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)

    print(s.getvalue())
    
import itertools
import time

# find the optimal parameters for the model
def optimise_parameters(module):
    # 1. Define the grid of parameters to test
    w_values = [0.3, 0.4, 0.5, 0.7]
    v_max_values = [0.02, 0.05, 0.10, 0.15]
    
    # Lock the particle count to a reliable number for DPSO
    particle_count = 5 

    # 2. Generate fixed test scenarios to ensure a fair test
    print("Generating Test Scenarios...")
    test_scenarios = []
    for _ in range(3): # Testing across 3 different random shading scenarios
        temps = [random.randint(25, 65) for _ in range(module.Ns)]
        irrs = [random.randint(100, 1000) for _ in range(module.Ns)]

        # calculate true peak to compare against
        module.set_cell_conditions(temps, irrs)
        _, _, powers = module.calculate_iv()
        true_pmp = powers[np.argmax(powers)]
        test_scenarios.append({'temps': temps, 'irrs': irrs, 'true_pmp': true_pmp})

    # 3. Test all different combinations
    best_config = None
    best_score = float('inf')

    print(f"{'w':<5} | {'v_max':<7} | {'Avg Error (W)':<15} | {'Time (s)':<10} | {'Score'}")
    print("-" * 60)

    for w, v_max in itertools.product(w_values, v_max_values):
        total_error = 0

        # create new tracker
        tracker = DPSO_MPPT(particle_count, module, 0, module.voc)
        
        # Override the tuning parameters
        tracker.w = w
        tracker.v_max = v_max

        # track time 
        start_time = time.time()
        for scenario in test_scenarios:
            tracker.set_module_conditions(scenario['temps'], scenario['irrs'])
            _, tracker_pmp, _ = tracker.global_optimisation()

            # get absolute error
            error = abs(tracker_pmp - scenario['true_pmp'])
            total_error += error

        elapsed = time.time() - start_time
        avg_error = total_error / len(test_scenarios)

        # score - want to penalize error heavily (e.g. 10x multiplier) while factoring in time
        score = (avg_error * 10) + elapsed 
        
        print(f"{w:<5} | {v_max:<7} | {avg_error:<15.4f} | {elapsed:<10.2f} | {score:.2f}")
        
        if score < best_score:
            best_score = score
            best_config = (w, v_max)

    print("-" * 60)
    print(f"BEST CONFIGURATION FOUND:")
    print(f"Inertia Weight (w): {best_config[0]}")
    print(f"Max Velocity (v_max): {best_config[1]}")
    print(f"Best Score: {best_score:.2f}")

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