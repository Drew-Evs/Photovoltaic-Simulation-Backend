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

        #whether tracking globally or locally p&o variables
        self.state = 'Global'
        self.po_step_size = 0.01
        self.po_direction = 1
        self.last_po_power = 0
        self.last_po_pos = 0
        self.power_drop_threshold = 0.10

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
        out = self.v_out*(D/(1-D))
        return np.clip(out, 0, self.module.voc)
    
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
    def set_module_conditions(self, temp_array=None, irr_array=None):
        self.module.set_cell_conditions(temp_array, irr_array)

    #run solver for a single duty cycle value
    def evaluate_single_position(self, D):
        voltage = self.get_voltage(D)
        
        current_guess = self.module.isc * 0.8
        guess = np.concatenate([
            [voltage/self.module.Ns]*self.module.Ns, [current_guess]*self.module.Ns,
            [0.0]*self.module.d, [0.0]*self.module.d, [current_guess]
        ])

        low_bound = np.array([-10.0]*self.module.Ns + [-np.inf]*(self.module.Ns + 2*self.module.d) + [0])
        low_bound[self.module.Ns:2*self.module.Ns] = 0
        high_bound = np.array([np.inf]*(2*self.module.Ns + 2*self.module.d + 1))
        high_bound[0:self.module.Ns] = [self.module.voc_per_cell]*self.module.Ns
        bounds = (low_bound, high_bound)

        power, _ = self.module.PSO_method(voltage, guess, bounds)
        return power

    #hybride tracker - tracks either globally if theres a large change in power or locally if minimal
    def track_mpp(self):
        #want to log this in a text file
        log_file = "power_over_time.txt"

        #runs the global optimisation method if in DPSO state
        if self.state == 'Global':
            with open(log_file, "a") as f:
                f.write("Large power change detected. Running Global DPSO Search\n")
 
            best_v, best_p, history = self.global_optimisation()

            #lock results into last p&o vars
            self.last_po_pos = self.gbest_position
            self.last_po_power = best_p

            #switch to P&O for future
            #self.state = 'Local'
            with open(log_file, "a") as f:
                f.write(f"Global Pmp: {self.last_po_pos:.3f}, Power {self.last_po_power:.2f}W\n")

            self.state = 'Local'

            return best_v, best_p, history

        elif self.state == 'Local':
            #take a small step 
            new_position = self.last_po_pos - (self.po_direction*self.po_step_size)
            new_position = np.clip(new_position, self.d_min, self.d_max)

            #evaluate at the new position
            new_power = self.evaluate_single_position(new_position)

            #prevent division by 0 and find absolute change in power
            safe_baseline = max(self.last_po_power, 0.1) 
            power_change_ratio = abs(new_power - safe_baseline) / safe_baseline

            '''currently triggers too often at the low level need to fix'''
            #test for a large drop (around 10%) - if so need to do global again (or also large increase)
            if power_change_ratio > self.power_drop_threshold:
                print(f'Power change of {power_change_ratio}W therefore global mode')
                self.state ='Global'
                return self.track_mpp()
                
            #if power goes down step in reverse direction
            if new_power < self.last_po_power:
                self.po_direction *= -1

            #update state
            self.last_po_power = new_power
            self.last_po_pos = new_position

            with open(log_file, "a") as f:
                f.write(f"P&O Tracking: Pos {new_position:.3f}, Power {new_power:.2f}W\n")

            return self.get_voltage(self.last_po_pos), self.last_po_power, [{'voltages': [self.get_voltage(self.last_po_pos)], 'powers': [self.last_po_power]}]
