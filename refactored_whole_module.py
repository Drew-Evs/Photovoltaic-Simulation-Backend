from predicting_parameters.refactored_single_cell import Cell
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import least_squares
import pvlib

#attempt to speed up using sparse matrix
from scipy.sparse import lil_matrix

import os

import random

#a module holding a series of cells - initate all at stcs
class Module():
    def __init__(self, datasheet_conditions, module_name):
        self.isc, self.vmp, self.voc, self.imp, self.Ns = datasheet_conditions
        self.voc_per_cell = self.voc/self.Ns
        self.cell_list = []

        for i in range(self.Ns):
            self.cell_list.append(Cell(1000, 25, datasheet_conditions, module_name))

        self.d = 3
        self.num_shaded = 0
        self.update_cell_arrays()

    #need to build the arrays for calculation 
    def update_cell_arrays(self):
        self.rs_arr = np.array([c.rs for c in self.cell_list])
        self.a_arr = np.array([c.a for c in self.cell_list])
        self.isat_arr = np.array([c.isat for c in self.cell_list])
        self.rsh_arr = np.array([c.rsh for c in self.cell_list])
        self.iph_arr = np.array([c.iph for c in self.cell_list])

    def voltage_residuals(self, x, voltage_load):
        k = 1.38e-23
        q = 1.6e-19

        #need the number of cells, number of diodes and the number of cells per diode
        c = self.Ns
        d = self.d
        p = c//d

        #gets the current values of voltage and current from x (cells and bypass diodes)
        v_c = x[0:c]
        i_c = x[c:2*c]
        v_bd = x[2*c:2*c+d]
        i_bd = x[2*c+d:2*c+2*d]
        i_panel = x[-1]

        #reshape into 2D grids
        v_c_2d = v_c.reshape((d, p))
        i_c_2d = i_c.reshape((d, p))

        #eq 7 load voltage
        eq7 = np.array([(np.sum(v_c) - voltage_load) * 10])

        #eq 8 mesh equations for each bd loop
        #the voltage in the byass diode should be opposite to that in the cells
        eq8 = (v_bd + np.sum(v_c_2d, axis=1)) * 10

        #eq 9 - each cell should have the same current inside bd loops
        #and eq 10/11 current of the panel is equal to each cell plus bypass diode
        eq9 = ((i_c_2d[:, :-1] - i_c_2d[:, 1:]) * 15).flatten()

        #eq 12 single cell current voltage relation
        #using constant values for boltzmans and electrical charge
        eq10 = i_panel - i_c_2d[:, 0] - i_bd

        # Calculate all 48 cell residuals simultaneously and add to the list
        exponent = (v_c + i_c * self.rs_arr) / self.a_arr
        exp_term = self.isat_arr * np.exp(np.clip(exponent, -50, 50))
        rsh_term = (v_c + i_c * self.rs_arr) / self.rsh_arr
        eq12 = -i_c + self.iph_arr - exp_term - rsh_term

        #eq 13 same for the bypass diodes
        #using constants for saturation current ideality and temperature
        i_sbd = 1.6e-9
        t_bd = 308.5
        arg_bd = np.clip((q * v_bd) / (1 * k * t_bd), -50, 50)
        eq13 = -i_bd + i_sbd * (np.exp(arg_bd) - 1)

        # Combine all residuals efficiently
        return np.concatenate([eq7, eq8, eq9, eq10, eq12, eq13])
    
    ##used in the MPPT to calculate an individual current for a voltage
    def PSO_method(self, voltage, initial_guess, bounds):
        solution = least_squares(
                self.voltage_residuals, 
                initial_guess, bounds=bounds,
                args=(voltage,)
            )

        #calculate the power for suitability - also update the initial guess for output
        x1 = solution.x

        v_c = solution.x[0:self.Ns]
        voltage = float(np.sum(v_c))
        i_panel = float(solution.x[-1])

        return voltage*i_panel, x1

    def calculate_iv(self):
        c = self.Ns
        d = self.d
        p = c//d

        #bounds for the cell
        low_bound = np.array([-10.0]*c + [-np.inf]*(c + d + d) + [0])
        low_bound[c:2*c] = 0
        high_bound = np.array([np.inf]*(c + c + d + d + 1))
        high_bound[0:c] = [self.voc_per_cell]*c

        #returning results
        powers = []
        currents = []
        voltages = []

        #initial guess at short circuit
        x0 = np.concatenate([
            [0.0]*c,               # Cell voltages near 0
            [self.isc]*c,          # Cell currents near short-circuit current
            [0.0]*d,               # Bypass diode voltages
            [0.0]*d,               # Bypass diode currents
            [self.isc]             # Total panel current
        ])

        i = 0

        #generating voltage loads to test
        voltage_targets = np.linspace(0, self.voc, 100)
        for voltage_target in voltage_targets:

            #print(f'Test number {i} at voltage {voltage_target}')
            i+= 1

            solution = least_squares(self.voltage_residuals, x0, bounds=(low_bound, high_bound),
                args=(voltage_target,))

            #getting the results
            v_c = solution.x[0:c]
            i_c = solution.x[c:2*c]
            v_bd = solution.x[2*c:2*c+d]
            i_bd = solution.x[2*c+d:2*c+2*d]
            i_panel = solution.x[-1]

            #update the guesses if correct
            if solution.success:
                x0 = solution.x
            else:
                print(f"Warning: Solver failed to converge at voltage {voltage_target}")

            #adding to graphs
            voltage = np.sum(v_c)
            power = voltage*i_panel        
            #power = np.sum(v_c*i_c)

            powers.append(power)
            voltages.append(voltage)
            currents.append(i_panel)

        return currents, voltages, powers
    
    #iterate through condition arrays
    def set_cell_conditions(self, temp_array=None, irr_array=None):
        if temp_array is None:
            temp_array = [cell.temperature for cell in self.cell_list]

        if irr_array is None:
            irr_array = [cell.irradiance for cell in self.cell_list]

        for i, (temp, irr) in enumerate(zip(temp_array, irr_array)):
            #test if change needed:
            if temp != self.cell_list[i].temperature or irr != self.cell_list[i].irradiance:
                #set cells then recalc array
                self.cell_list[i].shade(irr)
                self.cell_list[i].set_temp(temp)
                self.cell_list[i].predict_params()   
            
        #then create the module array
        self.update_cell_arrays()

    def refactored_iv(self):
        #generate a current sweep and preassign space for voltage sweep
        max_iph = max([c.iph for c in self.cell_list])
        I_sys = np.linspace(0, max_iph * 1.05, 50000) 
        V_module = np.zeros_like(I_sys)

        #split into substrings
        c_per_d = self.Ns // self.d
        substrings = [self.cell_list[i:i + c_per_d] for i in range(0, self.Ns, c_per_d)]

        #abalanche breakdown parameters
        VRBD = -5.5 #reverse breakdown voltag - when starts letting current flood backwards
        #control shape of curve
        aRBD = 1.036e-4 #avalanche fraction
        nRBD = 3.28 #avalanch exponent
        Vbypass = -0.5

        #find the voltage of substrings pre bypass activation
        for substr in substrings:
            V_sub_unbypassed = np.zeros_like(I_sys)

            #iterate across each substring cell calculating voltages for 
            for cell in substr:
                Vd_sweep = np.linspace(VRBD * 1.5, self.voc_per_cell * 1.2, 50000)
                bishop_multiplier = 1 + aRBD * (1 - Vd_sweep / VRBD)**(-nRBD)

                #explicit urrent solver
                exponent = np.clip(Vd_sweep / cell.a, -50, 50)
                I_cell_sweep = cell.iph - cell.isat * (np.exp(exponent) - 1) - (Vd_sweep / cell.rsh) * bishop_multiplier

                #convert to cell voltage
                V_terminal_sweep = Vd_sweep - I_cell_sweep * cell.rs

                #sort so both increasing and interpolate - matching unbypassed voltage to current
                sort_idx = np.argsort(I_cell_sweep)
                V_cell_interp = np.interp(I_sys, I_cell_sweep[sort_idx], V_terminal_sweep[sort_idx])
                V_sub_unbypassed += V_cell_interp

            #clamp voltage to bypass diode and add to whole mdoule voltaeg
            V_sub_bypassed = np.where(V_sub_unbypassed < Vbypass, Vbypass, V_sub_unbypassed)
            V_module += V_sub_bypassed

        #remove quadrant 1 (negaitve voltages)
        valid = V_module >= 0
        V_clean = V_module[valid]
        I_clean = I_sys[valid]

        return V_clean, V_clean * I_clean


def testing_curves(test_name, shaded_cells, shade_level):
    os.makedirs("module_graphs", exist_ok=True)
    os.makedirs(f'module_graphs/{test_name}', exist_ok=True)

    #get the datasheet conditions
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

    #create a module & graph it
    module = Module(datasheet_conditions, module_name)
    flat_shaded_cells = np.array(shaded_cells).flatten()
    module.num_shaded = len(flat_shaded_cells)

    #shade the parts that need shading
    for start, end in shaded_cells:
        print(f'Start is {start} and end is {end}')
        for i in range(start, end + 1):
            module.cell_list[i].shade(shade_level)

    #update the arrays used in the least square
    module.update_cell_arrays()

    currents, voltages, powers = module.calculate_iv(test_name)

    return currents, voltages, powers 

#for profiling
import cProfile
import pstats
import io

def run_profile():
    pr = cProfile.Profile()
    pr.enable()

    shaded_cells = np.array([[6,11], [43,47]])
    shade_level = 250
    currents, voltages, powers = testing_curves(shaded_cells, shade_level)

    max_power_point = powers[np.argmax(powers)]

    print(f"Max power point is {max_power_point}W")

    pr.disable()

    #sort values
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)

    print(s.getvalue())

if __name__ == "__main__":
    run_profile()


#     test_name = "1-2Cell_1-2Module_100W"

#     shaded_cells = np.array([])
#     shade_level = 100
#     currents, voltages, powers = testing_curves(test_name, shaded_cells, shade_level)

#     shaded_cells = np.array([[6,11], [43,47]])
#     shade_level = 250
#     currents1, voltages1, powers1 = testing_curves(test_name, shaded_cells, shade_level)

#     shaded_cells = np.array([[0,0], [16,16]])
#     shade_level = 100
#     currents2, voltages2, powers2 = testing_curves(test_name, shaded_cells, shade_level)

# # I–V Curve
#     plt.figure()
#     plt.plot(voltages, currents, label="No Shade")
#     plt.plot(voltages1, currents1, label="1-2 Module Shaded")
#     plt.plot(voltages2, currents2, label="1-2 Cell Shaded)")
#     plt.xlabel("Voltage (V)")
#     plt.ylabel("Current (A)")
#     plt.title("I–V Curve Comparison")
#     plt.grid(True)
#     plt.legend() # This adds the key to the graph
#     plt.savefig(f"module_graphs/{test_name}/iv_curve.png", dpi=300, bbox_inches="tight")
#     plt.close()

#     # P–V Curve
#     plt.figure()
#     plt.plot(voltages, powers, label="No Shade")
#     plt.plot(voltages1, powers1, label="1-2 Module Shaded")
#     plt.plot(voltages2, powers2, label="1-2 Cell Shaded")
#     plt.xlabel("Voltage (V)")
#     plt.ylabel("Power (W)")
#     plt.title("P–V Curve Comparison")
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(f"module_graphs/{test_name}/pv_curve.png", dpi=300, bbox_inches="tight")
#     plt.close()

#     # P–I Curve
#     plt.figure()
#     plt.plot(currents, powers, label="No Shade")
#     plt.plot(currents1, powers1, label="1-2 Module Shaded")
#     plt.plot(currents2, powers2, label="1-2 Cell Shaded")
#     plt.xlabel("Current (A)")
#     plt.ylabel("Power (W)")
#     plt.title("P–I Curve Comparison")
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(f"module_graphs/{test_name}/pi_curve.png", dpi=300, bbox_inches="tight")
#     plt.close()