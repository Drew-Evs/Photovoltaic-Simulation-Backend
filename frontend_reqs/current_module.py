from refactored_single_cell import Cell

import numpy as np
from scipy.optimize import fsolve

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
    
    def PSO_method(self, current, voltage_guess):
        # Diode physics constants
        k = 1.38e-23
        q = 1.6e-19
        i_sbd = 1.6e-9
        t_bd = 308.5
        v_thermal_bd = (1 * k * t_bd) / q

        cells_per_diode = self.Ns//self.d
        total_voltage = 0

        #run for each substring 
        for i in range(self.d):
            start_idx = i * cells_per_diode
            end_idx = start_idx + cells_per_diode
            substring_cells = self.cell_list[start_idx:end_idx]

            #calc voltage of substring
            v_sub = 0
            for cell in substring_cells:
                v_sub += fsolve(cell.iv_equation, voltage_guess/self.Ns, args=(current,))[0]

           # print(f"V_sub was {v_sub}")

            v_diode = -v_sub
            arg = np.clip(v_diode / v_thermal_bd, -50, 50)
            i_diode = i_sbd * (np.exp(arg) - 1.0)
            sub_current = current-i_diode
            sub_current = max(0.0, sub_current)

            #print(f"Current is now {sub_current}")

            v_sub = 0
            for cell in substring_cells:
                v_sub += fsolve(cell.iv_equation, voltage_guess/self.Ns, args=(sub_current,))[0]

            # print(f"New v_sub is {v_sub}")
            # input()

            total_voltage += v_sub

        return total_voltage * current 
      
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