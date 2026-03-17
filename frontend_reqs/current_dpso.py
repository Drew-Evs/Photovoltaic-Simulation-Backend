from current_module import Module
import concurrent.futures
import numpy as np

# '''as a class to make MPPT tracking simpler'''
# class C_DPSO_MPPT:
#     def __init__(self, num_particles, module, i_max):
#         #the module to track the max power of
#         self.module = module
#         self.v_out = module.voc
#         self.i_max = i_max

#         self.num_particles = num_particles

#         #initiate duty cycles for each particle and their velocity
#         self.particles = np.linspace(0, self.i_max, self.num_particles)
#         self.velocities = np.zeros(self.num_particles)

#         #tracck population personal bests and global best
#         self.pbest_positions = np.copy(self.particles)
#         self.pbest_powers =  np.zeros(self.num_particles)
#         self.gbest_position = self.particles[0]
#         self.gbest_power = 0

#         #just an inertia weight
#         self.w = 0.3
#         self.v_max = 0.05 

#         #whether tracking globally or locally p&o variables
#         self.state = 'Global'
#         self.po_step_size = 0.01
#         self.po_direction = 1
#         self.last_po_power = 0
#         self.last_po_pos = 0
#         self.power_drop_threshold = 0.10
    
#     #find the best duty cycle position
#     def global_optimisation(self):
#         #reset for new tracking
#         self.particles = np.linspace(0, self.i_max, self.num_particles)
#         self.velocities = np.zeros(self.num_particles)
#         self.pbest_positions = np.copy(self.particles)
#         self.pbest_powers =  np.zeros(self.num_particles)
#         self.gbest_position = self.particles[0]
#         self.gbest_power = 0

#         #using a tolerance to end loop
#         max_iter = 100
#         iter = 0
#         tol = 1e-4

#         #skip if power not meaningfully changing
#         power_tol = 1e-6
#         #if voltage barely changes then want to skip rerunning solver
#         current_threshold = 0.05

#         #store powers/force initial run
#         powers = np.zeros(self.num_particles)
#         previous_currents = np.zeros(self.num_particles) - 999
#         stall_counter = 0

#         #optimise series of steps
#         while iter < max_iter:
#             #update powers and guesses by previous result
#             #add to queue for parallelisation
#             with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_particles) as executor:
#                 future_to_idx = {}
#                 for j, I in enumerate(self.particles):
#                     #only update if voltage changes significantly
#                     if abs(I-previous_currents[j]) > current_threshold:
#                         #submit to solver as a background thread 
#                         voltage_guess = self.module.voc * max(0.0, (1.0 - (I / self.i_max)))
#                         future = executor.submit(self.module.PSO_method, I, voltage_guess)
#                         future_to_idx[future] = j

#                 #collect results as they finish
#                 for future in concurrent.futures.as_completed(future_to_idx):
#                     i = future_to_idx[future]
#                     power = future.result()
#                     powers[i] = power

#             previous_currents = np.copy(self.particles)

#             #store to check for power stalling before optimising
#             old_gbest_power = self.gbest_power
#             self.optimise_step(powers)

#             #early stop check
#             #if variance between all particle positions are basically 0
#             if np.std(self.particles) < tol:
#                 print(f'Converged after {iter} iterations')
#                 break
            
#             #increase stall if iteration not improved
#             if (self.gbest_power - old_gbest_power) < power_tol:
#                 stall_counter += 1
#                 #power needs to stay the same for 3 loops
#                 if stall_counter >= 4:
#                     print(f'Converged on power after {iter} iterations.')
#                     break
#             else:
#                 stall_counter = 0

#             iter += 1

#             if iter == max_iter:
#                 print(f'Didnt converge within max iterations')

#         return self.gbest_position, self.gbest_power

#     #each individual step of the optimisation
#     def optimise_step(self, current_powers):
#         #global dpso 
#         #changed to vector and masking to improve speed
#         mask = current_powers > self.pbest_powers
#         self.pbest_powers[mask] = current_powers[mask]
#         self.pbest_positions[mask] = self.particles[mask]

#         #global evaluation
#         best_idx = np.argmax(current_powers)
#         if current_powers[best_idx] > self.gbest_power:
#             self.gbest_power = current_powers[best_idx]
#             self.gbest_position = self.particles[best_idx]

#         #updated velocity calculation - and clipping
#         self.velocities = (self.w*self.velocities) + (self.gbest_position + self.pbest_positions - 2*self.particles)
#         self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
#         self.particles += self.velocities

#         #vectorised clipping
#         self.particles = np.clip(self.particles, 0, self.i_max)

#     #iterate through condition arrays
#     def set_module_conditions(self, temp_array=None, irr_array=None):
#         self.module.set_cell_conditions(temp_array, irr_array)

#     def evaluate_single_position(self, I):
#         # Dynamic guess: If current is high, guess low voltage. If low, guess high voltage.
#         voltage_guess = self.module.voc * max(0.0, (1.0 - (I / self.i_max)))

#         power = self.module.PSO_method(I, voltage_guess)
#         return power

#     #hybride tracker - tracks either globally if theres a large change in power or locally if minimal
#     def track_mpp(self):

#         #runs the global optimisation method if in DPSO state
#         if self.state == 'Global':
#             best_i, best_p= self.global_optimisation()

#             #lock results into last p&o vars
#             self.last_po_pos = self.gbest_position
#             self.last_po_power = best_p

#             #switch to P&O for future
#             self.state = 'Local'

#             return best_i, best_p

#         elif self.state == 'Local':
#             #take a small step 
#             new_position = self.last_po_pos + (self.po_direction*self.po_step_size)
#             new_position = np.clip(new_position, 0, self.i_max)

#             #evaluate at the new position
#             new_power = self.evaluate_single_position(new_position)

#             #prevent division by 0 and find absolute change in power
#             safe_baseline = max(self.last_po_power, 0.1) 
#             power_change_ratio = abs(new_power - safe_baseline) / safe_baseline

#             #test for a large drop (around 10%) - if so need to do global again (or also large increase)
#             if power_change_ratio > self.power_drop_threshold:
#                 self.state ='Global'
#                 return self.track_mpp()
                
#             #if power goes down step in reverse direction
#             if new_power < self.last_po_power:
#                 self.po_direction *= -1

#             #update state
#             self.last_po_power = new_power
#             self.last_po_pos = new_position

#         return self.last_po_pos, self.last_po_power


import numpy as np

class C_DPSO_MPPT:
    def __init__(self, num_particles, module, i_max):
        self.module = module
        self.i_max = i_max
        self.num_points = 30 # Hardcoded to test exactly 30 points
        
        # State just kept so the print statement in dpso_comparison.py doesn't crash
        self.state = 'Brute-Force Sweep' 

    def set_module_conditions(self, temp_array=None, irr_array=None):
        self.module.set_cell_conditions(temp_array, irr_array)

    def track_mpp(self):
        best_power = 0.0
        best_current = 0.0
        
        # Generate exactly 30 currents from 0 A to Short Circuit Current
        test_currents = np.linspace(0, self.i_max, self.num_points)
        
        for current in test_currents:
            # Dynamic guess: If current is high, guess low voltage. If low, guess high voltage.
            voltage_guess = self.module.voc * max(0.0, (1.0 - (current / self.i_max)))
            
            # Get the power from the module's fsolve method
            power = self.module.PSO_method(current, voltage_guess)
            
            # Keep track of the highest power found
            if power > best_power:
                best_power = power
                best_current = current

        # Return the best current and its corresponding power
        return best_current, best_power