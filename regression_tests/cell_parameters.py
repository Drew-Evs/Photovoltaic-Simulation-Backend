#importing from a different folder
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from predicting_parameters import refactored_prediction

import pvlib
import pytest
import numpy as np

cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
module = cec_modules['Canadian_Solar_Inc__CS1U_405MS']

#different conditions to test
TEST_CONDITIONS = [
    (25, 1000),
    (10, 200),   
    (45, 800), 
    (60, 1000), 
    (25, 500),
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
def test_Iph(parameter_sets):
    pv_val, ref_val = parameter_sets['pvlib'][0], parameter_sets['model'][0]
    cond = parameter_sets['condition']
    assert np.isclose(ref_val, pv_val, rtol=0.001), f"Iph at {cond}: got {ref_val}, expected {pv_val}"

def test_Isat(parameter_sets):
    pv_val, ref_val = parameter_sets['pvlib'][1], parameter_sets['model'][1]
    cond = parameter_sets['condition']
    assert np.isclose(ref_val, pv_val, rtol=0.01), f"Isat at {cond}: got {ref_val}, expected {pv_val}"

def test_Rs(parameter_sets):
    pv_val, ref_val = parameter_sets['pvlib'][2], parameter_sets['model'][2]
    cond = parameter_sets['condition']
    assert np.isclose(ref_val, pv_val, rtol=0.01), f"Rs at {cond}: got {ref_val}, expected {pv_val}"

def test_Rsh(parameter_sets):
    pv_val, ref_val = parameter_sets['pvlib'][3], parameter_sets['model'][3]
    cond = parameter_sets['condition']
    assert np.isclose(ref_val, pv_val, rtol=0.07), f"Rsh at {cond}: got {ref_val}, expected {pv_val}"

def test_a(parameter_sets):
    pv_val, ref_val = parameter_sets['pvlib'][4], parameter_sets['model'][4]
    cond = parameter_sets['condition']
    assert np.isclose(ref_val, pv_val, rtol=0.01), f"a at {cond}: got {ref_val}, expected {pv_val}"
