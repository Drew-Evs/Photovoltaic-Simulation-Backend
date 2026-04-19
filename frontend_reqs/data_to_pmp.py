import socket
import numpy as np
import time
import select
import pandas as pd

import pvlib
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from power_tracking.refactored_whole_module import Module
from power_tracking.DPSO_MPPT import DPSO_MPPT

def run_shade_to_pmp(module, input_csv, output_csv):
    #create the tracker
    tracker =  DPSO_MPPT(module.d, module, 0, module.voc)
    tracker.state = 'Global'

    data = pd.read_csv(input_csv)

    output = []

    #get irradiance list
    for i, row in enumerate(data.iloc[:, :].to_numpy()):
        #number of shaded bypass diodes substrings
        #keep tracker working
        tracker.set_module_conditions(irr_array=row)
        avg_irr = np.mean(row)
        if avg_irr == 100:
            pmp = 0
        else:
            _, pmp = tracker.track_mpp()

        shaded_subs = module.count_shaded_substrings(row)
        
        #calculate the time (based on 450 items meaning 24 hours)
        elapsed_time = 192*i

        output.append({"time": elapsed_time, "power": pmp, "shaded_substrings": shaded_subs})
        tracker.state = 'Local'

    df = pd.DataFrame(output)
    df.to_csv(output_csv, index=False)

#new model just using module
def run_shade_to_pmp_new(module, input_csv, output_csv):
    data = pd.read_csv(input_csv)
    output = []

    #get irradiance list
    for i, row in enumerate(data.iloc[:, :].to_numpy()):
        #number of shaded bypass diodes substrings
        #keep tracker working
        avg_irr = np.mean(row)
        if avg_irr > 0:
            module.set_cell_conditions(irr_array=row)
            voltages, powers = module.refactored_iv()
        else:
            pmp = 0
        
        shaded_subs = module.count_shaded_substrings(row)
        
        #calculate the time (based on 450 items meaning 24 hours)
        elapsed_time = 192*i
        print(f'Step {i}')

        output.append({"time": elapsed_time, "power": np.max(powers), "shaded_substrings": shaded_subs})

    df = pd.DataFrame(output)
    df.to_csv(output_csv, index=False)

#trial run using the pvmismatch library to test
def pvmismatch_test(module, module_pvmm, input_csv, output_csv):
    data = pd.read_csv(input_csv)
    output = []

    #get irradiance list
    for i, row in enumerate(data.iloc[:, :].to_numpy()):
        #number of shaded bypass diodes substrings
        #keep tracker working
        avg_irr = np.mean(row)
        if avg_irr > 0.0:
            raw_suns = np.array(row) / 1000.0
            suns_array = np.clip(raw_suns, 0.001, None)
            module_pvmm.setSuns(suns_array)
            _, _, Pmod, _, _ = module_pvmm.calcMod()
            P_pvmm = Pmod.flatten()
            valid_P = P_pvmm[~np.isnan(P_pvmm)]

            if len(valid_P) > 0:
                pmp = np.max(valid_P)
            else:
                pmp = 0
        else:
            pmp = 0
        
        shaded_subs = module.count_shaded_substrings(row)
        
        #calculate the time (based on 450 items meaning 24 hours)
        elapsed_time = 192*i

        output.append({"time": elapsed_time, "power": pmp, "shaded_substrings": shaded_subs})

    df = pd.DataFrame(output)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    from pvmismatch import pvsystem
    from pvmismatch.pvmismatch_lib import pvmodule, pvstring, pvcell
    from pvmismatch.pvmismatch_lib.pvconstants import PVconstants

    #hold data in a folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(BASE_DIR, "simulation_data")

    #ensure folder exists
    os.makedirs(target_dir, exist_ok=True)

    #finds input and output file
    input_csv = os.path.join(target_dir, "Saved_Irradiance_Data.csv")
    output_csv = os.path.join(target_dir, "Saved_Power_Time_Data.csv")

    #ensures input exists
    if not os.path.exists(input_csv):
        print(f"Error: Input data not found at {input_csv}")
    else:
        cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
        module = cec_modules['Prism_Solar_Technologies_Bi48_267BSTC']
        module_name = 'Prism_Solar_Technologies_Bi48_267BSTC'
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
        
        module = Module('Prism_Solar_Technologies_Bi48_267BSTC', specs)

        run_shade_to_pmp_new(module, input_csv, output_csv)

        # #create the pvmismatch version
        # custom_const = PVconstants()
        # a, iph, isat, rs, rsh = module.cell_list[0].get_params()
        # T = 298.15 
        # Vt_standard = (custom_const.k * T) / custom_const.q
        # n_ideal = a / Vt_standard
        # custom_const.k = custom_const.k * n_ideal

        # lg_cells = [
        #     pvcell.PVcell(Rs=rs, Rsh=rsh, Isat1_T0=isat, Isat2_T0=0, Isc0_T0=iph, pvconst=custom_const, alpha_Isc=specs['I_sc']) 
        #     for _ in range(48)
        # ]

        # custom_layout = pvmodule.standard_cellpos_pat(nrows=8, ncols_per_substr=[2, 2, 2])
        # panel_48 = pvmodule.PVmodule(cell_pos=custom_layout, pvcells=lg_cells)
        # panel_string = pvstring.PVstring(numberMods=1, pvmods=[panel_48])
        # pvmm_sys = pvsystem.PVsystem(numberStrs=1, pvstrs=[panel_string])

        # pvmismatch_test(module, pvmm_sys, input_csv, output_csv)