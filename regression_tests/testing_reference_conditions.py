#importing from a different folder
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from predicting_parameters import reference_conditions

import pvlib
import pytest
import numpy as np

cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
module = cec_modules['Canadian_Solar_Inc__CS1U_405MS']
print(module)

#reference thermal voltage
Ns = module['N_s']
k = 1.380649e-23
q = 1.602176634e-19
NsVth = k*298.15*Ns/q

print(f'Reference isat = {module['I_o_ref']}')

#reference coditinos from pv lib
pv_iph, pv_isat, pv_Rs, pv_Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
    effective_irradiance=1000,
    temp_cell=25,
    alpha_sc=module['alpha_sc'],
    a_ref=module['a_ref'],
    I_L_ref=module['I_L_ref'],
    I_o_ref=module['I_o_ref'],
    R_sh_ref=module['R_sh_ref'],
    R_s=module['R_s'],
    EgRef=1.121,
    dEgdT=-0.0002677
)

pv_n = nNsVth/NsVth

print(
    f"Iph  = {pv_iph:.4f} A\n"
    f"Isat = {pv_isat:.3e} A\n"
    f"Rs   = {pv_Rs:.4f} Ω\n"
    f"Rsh  = {pv_Rsh:.2f} Ω\n"
    f"n    = {pv_n:.3f}"
)

# #using ref conditions code
Iph_ref, Isat_ref, Rs_ref, Rsh_ref, n_ref = reference_conditions.get_reference_params()

#regression tests
def test_Iph():
    assert np.isclose(Iph_ref, pv_iph, atol=1e-2), f"Iph: got {Iph_ref}, expected {pv_iph}"

def test_Isat():
    assert np.isclose(Isat_ref, pv_isat, rtol=1e-2), f"Isat: got {Isat_ref}, expected {pv_isat}"

def test_Rs():
    assert np.isclose(Rs_ref, pv_Rs, atol=1e-4), f"Rs: got {Rs_ref}, expected {pv_Rs}"

def test_Rsh():
    assert np.isclose(Rsh_ref, pv_Rsh, atol=1e-1), f"Rsh: got {Rsh_ref}, expected {pv_Rsh}"

def test_n():
    assert np.isclose(n_ref, pv_n, atol=1e-2), f"n: got {n_ref}, expected {pv_n}"
