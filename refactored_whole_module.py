from predicting_parameters.refactored_single_cell import Cell
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import least_squares
import pvlib

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

        #stores the residuals
        res = [] 

        #eq 7 load voltage
        res.append(np.sum(v_c) - voltage_load)

        #eq 8 mesh equations for each bd loop
        #the voltage in the byass diode should be opposite to that in the cells
        for i in range(d):
            start, end = i*p, (i+1)*p
            res.append((v_bd[i] + np.sum(v_c[start:end]))*10)

        #eq 9 - each cell should have the same current inside bd loops
        #and eq 10/11 current of the panel is equal to each cell plus bypass diode
        for bd in range(d):
            start = bd * p
            end = (bd + 1) * p
            for i in range(start, end - 1):
                res.append(i_c[i] - i_c[i + 1]) 

            res.append(i_panel - i_c[start] - i_bd[bd])

        #eq 12 single cell current voltage relation
        #using constant values for boltzmans and electrical charge
        exponent = (v_c + i_c * self.rs_arr) / self.a_arr
        exp_term = self.isat_arr * np.exp(np.clip(exponent, -50, 50))
        rsh_term = (v_c + i_c * self.rs_arr) / self.rsh_arr

        # Calculate all 48 cell residuals simultaneously and add to the list
        res.extend((-i_c + self.iph_arr - exp_term - rsh_term).tolist())

        #eq 13 same for the bypass diodes
        #using constants for saturation current ideality and temperature
        i_sbd = 1.6e-9
        n_bd = 1
        t_bd = 308.5 #(kelvin)
        for i in range(d):
            arg_bd = np.clip((q * v_bd[i]) / (n_bd * k * t_bd), -50, 50)
            res.append(-i_bd[i] + i_sbd * (np.exp(arg_bd) - 1))

        return res

    def calculate_iv(self, test_name):
        self.rs_arr = np.array([c.rs for c in self.cell_list])
        self.a_arr = np.array([c.a for c in self.cell_list])
        self.isat_arr = np.array([c.isat for c in self.cell_list])
        self.rsh_arr = np.array([c.rsh for c in self.cell_list])
        self.iph_arr = np.array([c.iph for c in self.cell_list])

        c = self.Ns
        d = self.d

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
        voltage_targets = np.linspace(0, self.voc, 70)
        for voltage_target in voltage_targets:

            print(f'Test number {i} at voltage {voltage_target}')
            i+= 1

            solution = least_squares(self.voltage_residuals, x0, bounds=(low_bound, high_bound),
                args=(voltage_target,))

            #getting the results
            v_c = solution.x[0:c]
            i_c = solution.x[c:2*c]
            v_bd = solution.x[2*c:2*c+d]
            i_bd = solution.x[2*c+d:2*c+2*d]
            i_panel = solution.x[-1]

            #adding to graphs
            voltage = np.sum(v_c)
            power = voltage * i_panel

            powers.append(power)
            voltages.append(voltage)
            currents.append(i_panel)

            #update the guesses if correct
            if solution.success:
                x0 = solution.x
            else:
                print(f"Warning: Solver failed to converge at voltage {voltage_target}")

        #plot
        plt.figure()
        plt.plot(voltages, currents)
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title("I–V Curve")
        plt.grid(True)
        plt.savefig(f"module_graphs/{test_name}/iv_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

        # P–V
        plt.figure()
        plt.plot(voltages, powers)
        plt.xlabel("Voltage (V)")
        plt.ylabel("Power (W)")
        plt.title("P–V Curve")
        plt.grid(True)
        plt.savefig(f"module_graphs/{test_name}/pv_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

        # P–I
        plt.figure()
        plt.plot(currents, powers)
        plt.xlabel("Current (A)")
        plt.ylabel("Power (W)")
        plt.title("P–I Curve")
        plt.grid(True)
        plt.savefig(f"module_graphs/{test_name}/pi_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

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
        for i in range(start, end + 1):
            module.cell_list[i].shade(shade_level)

    module.calculate_iv(test_name)
    

if __name__ == "__main__":
    shaded_cells = np.array([[5,11],[42,47]])
    shade_level = 250
    test_name = "6-12,43-48Cell_250W"
    testing_curves(test_name, shaded_cells, shade_level)