#importing from a different folder
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from predicting_parameters import refactored_prediction
from predicting_parameters import refactored_single_cell
from predicting_parameters import cell_ann

import pvlib
import pytest
import numpy as np

cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
module = cec_modules['Canadian_Solar_Inc__CS1U_405MS']

#different conditions to test
TEST_CONDITIONS = [
    (25, 1000),
    (45, 1000),   
    (65, 1000), 
    (25, 600),
    (45, 600),   
    (65, 600),
    (25, 200),
    (45, 200),   
    (65, 200),  
]

#specs from datasheet imported from pvlibs
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

datasheet_conditions = [module['I_sc_ref'], module['V_mp_ref'], module['V_oc_ref'], module['I_mp_ref'], module['N_s']]

@pytest.fixture(params=TEST_CONDITIONS, ids=lambda x: f"T={x[0]}C,Irr={x[1]}")
def parameter_sets(request):
    t_c, irr = request.param

    #reference parameters from pv lib
    pv_iph, pv_isat, pv_Rs, pv_Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
        effective_irradiance=irr,
        temp_cell=t_c,
        alpha_sc=module['alpha_sc'],
        a_ref=module['a_ref'],
        I_L_ref=module['I_L_ref'],
        I_o_ref=module['I_o_ref'],
        R_sh_ref=module['R_sh_ref'],
        R_s=module['R_s'],
        EgRef=1.121,
        dEgdT=-0.0002677
    )

    #predicting at conditions
    Iph_ref, Isat_ref, Rs_ref, Rsh_ref, a_ref = refactored_prediction.getting_parameters_specs(t_c, irr, specs)

    #return dict values
    return {
        'pvlib': (pv_iph, pv_isat, pv_Rs, pv_Rsh, nNsVth),
        'model': (Iph_ref, Isat_ref, Rs_ref, Rsh_ref, a_ref),
        'condition': f"T={t_c}C, Irr={irr}W/m^2"
    } 

#regression tests
# def test_analytical_parameters(parameter_sets, subtests):
#     #unpack to tuples
#     pv_vals = parameter_sets['pvlib']
#     ref_vals = parameter_sets['model']
#     cond = parameter_sets['condition']

#     #define param names
#     checks = [
#         ("Iph", 0, 0.001),
#         ("Isat", 1, 0.01),
#         ("Rs", 2, 0.01),
#         ("Rsh", 3, 0.07),
#         ("a", 4, 0.01)
#     ]

#     #run subtest for each param at each conditoin
#     for param_name, idx, tol in checks:
#         pv_val = pv_vals[idx]
#         ref_val = ref_vals[idx]

#         # print(f"{cond} | {param_name} | Expected (PVLib): {pv_val:.5g} | Got (Model): {ref_val:.5g}")
        
#         with subtests.test(msg=param_name):
#             assert np.isclose(ref_val, pv_val, rtol=tol), \
#                 f"{param_name} at {cond}: got {ref_val}, expected {pv_val}"

#create a single cell - using ANN and a data entry class to test against
#data entry is whats used to build training data - shown to be accurate
@pytest.fixture(scope="function")
def base_system():
    #initate at stc
    test_cell = refactored_single_cell.Cell(1000, 25, 'Canadian_Solar_Inc__CS1U_405MS', specs)
    return test_cell

@pytest.fixture(params=TEST_CONDITIONS, ids=lambda x: f"T={x[0]}C,Irr={x[1]}")
def parameter_sets_cell(request, base_system):
    test_cell = base_system

    t_c, irr = request.param

    test_cell.shade(irr)
    test_cell.set_temp(t_c)
    test_cell.predict_params()

    #then create the data entry with these conditions
    data_entry = cell_ann.DataEntry(irr, t_c, datasheet_conditions, 'Canadian_Solar_Inc__CS1U_405MS', specs)


    #return dict values
    return {
        'ann': test_cell.get_params(),
        'data_entry': data_entry.get_params(),
        'condition': f"T={t_c}C, Irr={irr}W/m^2"
    } 

#regression test cells
def test_ann_parameters(parameter_sets_cell, subtests):
    #get the ann values and reference values to unpack
    ann_vals = parameter_sets_cell['ann']
    ref_vals = parameter_sets_cell['data_entry']
    cond = parameter_sets_cell['condition']

    #define as parameters
    checks = [
        ("a", 0, {"rtol": 0.05}),
        ("Iph", 1, {"rtol": 0.1}),
        ("Isat", 2, {"rtol": 0.05}),
        ("Rs", 3, {"rtol": 0.05}),
        ("Rsh", 4, {"atol": 1e1})
    ]

    #test for each parameter
    for param_name, idx, tol_kwargs in checks:
        ann_val = ann_vals[idx]
        ref_val = ref_vals[idx]
        
        #format and output
        print(f"{cond} | {param_name} | Expected (DataEntry): {ref_val:.5g} | Got (ANN): {ann_val:.5g}")

        with subtests.test(msg=param_name):
            assert np.isclose(ref_val, ann_val, **tol_kwargs), \
                f"{param_name} at {cond}: got {ann_val}, expected {ref_val}"
