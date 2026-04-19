import socket
import numpy as np
import time
import select
import pandas as pd

import pvlib
from refactored_whole_module import Module

from DPSO_MPPT import DPSO_MPPT

def run_shade_to_pmp(module):
    #create the tracker
    tracker =  DPSO_MPPT(module.d, module, 0, module.voc)
    tracker.state = 'Global'

    data = pd.read_csv("Saved_Irradiance_Data.csv")
    log_file = "unity_power_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Step, Time, Unity Avg Irradiance, Power(W), Tracker State, Shaded Substrings\n")

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

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{i}, {elapsed_time:.2f}, {avg_irr:.0f}, {pmp:.2f}, {tracker.state}, {shaded_subs}\n")

        output.append({"time": elapsed_time, "power": pmp, "shaded_substrings": shaded_subs})
        tracker.state = 'Local'

    df = pd.DataFrame(output)
    df.to_csv("Saved_Power_Time_Data2.csv", index=False)

#new model just using module
def run_shade_to_pmp_new(module):
    data = pd.read_csv("Saved_Irradiance_Data.csv")
    log_file = "unity_power_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Step, Time, Unity Avg Irradiance, Power(W), Shaded Substrings\n")

    output = []

    #get irradiance list
    for i, row in enumerate(data.iloc[:, :].to_numpy()):
        #number of shaded bypass diodes substrings
        #keep tracker working
        avg_irr = np.mean(row)
        if avg_irr > 0:
            module.set_cell_conditions(irr_array=row)
            pmp = module.refactored_iv()
        else:
            pmp = 0
        
        shaded_subs = module.count_shaded_substrings(row)
        
        #calculate the time (based on 450 items meaning 24 hours)
        elapsed_time = 192*i

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{i}, {elapsed_time:.2f}, {avg_irr:.0f}, {pmp:.2f}, {shaded_subs}\n")

        output.append({"time": elapsed_time, "power": pmp, "shaded_substrings": shaded_subs})


    df = pd.DataFrame(output)
    df.to_csv("Saved_Power_Time_Data2.csv", index=False)

#trial run using the pvmismatch library to test
def pvmismatch_test(module, module_pvmm):
    data = pd.read_csv("Saved_Irradiance_Data.csv")
    log_file = "unity_power_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Step, Time, Unity Avg Irradiance, Power(W), Shaded Substrings\n")

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

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{i}, {elapsed_time:.2f}, {avg_irr:.0f}, {pmp:.2f}, {shaded_subs}\n")

        output.append({"time": elapsed_time, "power": pmp, "shaded_substrings": shaded_subs})


    df = pd.DataFrame(output)
    df.to_csv("Saved_Power_Time_Data2.csv", index=False)


if __name__ == "__main__":
    from pvmismatch import pvsystem
    from pvmismatch.pvmismatch_lib import pvmodule, pvstring, pvcell
    from pvmismatch.pvmismatch_lib.pvconstants import PVconstants

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

    #run_shade_to_pmp_new(module)

    #create the pvmismatch version
    custom_const = PVconstants()
    a, iph, isat, rs, rsh = module.cell_list[0].get_params()
    T = 298.15 
    Vt_standard = (custom_const.k * T) / custom_const.q
    n_ideal = a / Vt_standard
    custom_const.k = custom_const.k * n_ideal

    lg_cells = [
        pvcell.PVcell(Rs=rs, Rsh=rsh, Isat1_T0=isat, Isat2_T0=0, Isc0_T0=iph, pvconst=custom_const, alpha_Isc=datasheet_conditions[0]) 
        for _ in range(48)
    ]

    custom_layout = pvmodule.standard_cellpos_pat(nrows=8, ncols_per_substr=[2, 2, 2])
    panel_48 = pvmodule.PVmodule(cell_pos=custom_layout, pvcells=lg_cells)
    panel_string = pvstring.PVstring(numberMods=1, pvmods=[panel_48])
    pvmm_sys = pvsystem.PVsystem(numberStrs=1, pvstrs=[panel_string])

    pvmismatch_test(module, pvmm_sys)