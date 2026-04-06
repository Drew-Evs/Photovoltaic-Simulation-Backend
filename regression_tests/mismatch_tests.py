import sys
from pathlib import Path
import pytest
import numpy as np
import pvlib

from pvmismatch import pvsystem
from pvmismatch.pvmismatch_lib import pvmodule, pvstring, pvcell
from pvmismatch.pvmismatch_lib.pvconstants import PVconstants#

sys.path.append(str(Path(__file__).resolve().parents[1]))
import refactored_whole_module

#the test cases used in the demo
irr_case_1 = [1000.0] * 60

irr_case_2 = [1000.0] * 60
irr_case_2[0:29] = [100.0] * 29

irr_case_3 = [1000.0] * 60
irr_case_3[0:3] = [300.0] * 3
irr_case_3[31:34] = [800.0] * 3

irr_case_4 = [1000.0] * 60
irr_case_4[0:15] = [200.0] * 15
irr_case_4[31:45] = [300.0] * 14

irr_case_5 = [1000.0] * 60
irr_case_5[0:20] = [400.0] * 20
irr_case_5[20:40] = [600.0] * 20
irr_case_5[40:60] = [800.0] * 20

irr_case_6 = [1000.0] * 60
irr_case_6[10:11] = [50.0] * 1

irr_case_7 = [200.0] * 60

irr_case_8 = [1000.0] * 60
irr_case_8[0:1] = [10.0] * 1
irr_case_8[20:21] = [10.0] * 1
irr_case_8[40:41] = [10.0] * 1

TEST_CASES = [
    ("Unshaded (STC)", irr_case_1),
    ("One Substring Shaded", irr_case_2),
    ("Two Substrings Lightly Shaded", irr_case_3),
    ("Two Substrings Heavily shaded", irr_case_4),
    ("Three Substrings Gradient Shade", irr_case_5),
    ("Single Cell Heavy Shade", irr_case_6),
    ("Uniform Low Irradiance", irr_case_7),
    ("One Cell Shaded Per Substring", irr_case_8)
]

#creating the 2 modules
@pytest.fixture(scope="module")
def base_systems():
    #specs from the LG330N1T-V5 panel datasheet
    specs = {
        'tech': 'mono-c-si',
        'N_s': 60,
        'I_sc': 10.3,        
        'V_oc': 40.6,
        'I_mp': 9.77,
        'V_mp': 33.8,
        'alpha_sc': 0.03,
        'beta_oc': -0.27,
        'gamma': -0.36
    }

    #use specs for the custom module
    custom_mod = refactored_whole_module.Module("LG_Electronics_Inc__LG330N1T_V5", specs)
    custom_mod.d = 3

    #setup pvmismatch at stc 
    custom_const = PVconstants()
    a, iph, isat, rs, rsh = custom_mod.cell_list[0].get_params()
    T = 298.15 
    Vt_standard = (custom_const.k * T) / custom_const.q
    n_ideal = a / Vt_standard
    custom_const.k = custom_const.k * n_ideal

    lg_cells = [
        pvcell.PVcell(Rs=rs, Rsh=rsh, Isat1_T0=isat, Isat2_T0=0, Isc0_T0=iph, 
                      pvconst=custom_const, alpha_Isc=specs['I_sc']) 
        for _ in range(60)
    ]

    #create 60 cell 3 bypass diode layout
    custom_layout = pvmodule.standard_cellpos_pat(nrows=20, ncols_per_substr=[1, 1, 1])
    panel_60 = pvmodule.PVmodule(cell_pos=custom_layout, pvcells=lg_cells)
    panel_string = pvstring.PVstring(numberMods=1, pvmods=[panel_60])
    pvmm_sys = pvsystem.PVsystem(numberStrs=1, pvstrs=[panel_string])

    return custom_mod, pvmm_sys

@pytest.mark.parametrize("test_name, irr_array", TEST_CASES)
def test_pmp_shading_profiles(base_systems, test_name, irr_array):
    custom_mod, pvmm_sys = base_systems

    #benchmark with pvmismatch
    suns_array = np.array(irr_array) / 1000.0
    pvmm_sys.setSuns({0: {0: suns_array}})
    P_pvmm = pvmm_sys.Psys.flatten()
    benchmark_pmp = np.max(P_pvmm)

    #predict wth custom module
    custom_mod.set_cell_conditions(irr_array=irr_array)
    voltages, powers = custom_mod.calculate_iv()
    custom_pmp = np.max(powers)

    assert np.isclose(custom_pmp, benchmark_pmp, rtol=0.1), \
        f"{test_name} FAILED: Custom Pmp = {custom_pmp:.2f}W, Benchmark = {benchmark_pmp:.2f}W"