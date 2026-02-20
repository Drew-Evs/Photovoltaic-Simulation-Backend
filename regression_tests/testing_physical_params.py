import pvlib
import pytest
import numpy as np

#importing from a different folder
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from predicting_parameters import physical_params

#getting values for regression testing
cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
list(cec_modules.keys())[:10]
module = cec_modules['Canadian_Solar_Inc__CS6K_270M']
print(module)

#constants needed
Ns = module['N_s']
k = 1.380649e-23
q = 1.602176634e-19

#need reference temperature and irradiance
ref_temp = 25
ref_irradiance = 1000

#getting more refference conditions
Iph, Isat, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_cec(
    effective_irradiance=ref_irradiance,
    temp_cell=ref_temp,
    alpha_sc=module['alpha_sc'],
    a_ref=module['a_ref'],
    I_L_ref=module['I_L_ref'],
    I_o_ref=module['I_o_ref'],
    R_sh_ref=module['R_sh_ref'],
    R_s=module['R_s'],
    Adjust=module['Adjust']
)
#reference ideality 
thermal_voltage = k*298.15/q
ref_n = nNsVth/(Ns*thermal_voltage)
ref_rsh = Rsh
ref_isat = Isat

#in form (irradiance, real temperature)
test_cases = [
    (100, 25),
    (100, 45),
    (100, 65),

    (500, 25),
    (500, 45),
    (500, 65),

    (1000, 25),
    (1000, 45),
    (1000, 65),

    (300, 35),
    (700, 55),
    (900, 60),
]

# @pytest.mark.parametrize("irradiance, real_temp", test_cases)
# def test_ideality(irradiance, real_temp):
#     k_temp_real = real_temp+273.15
#     k_temp_ref = ref_temp+273.15
#     #get the expected values
#     Iph, Isat, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
#         effective_irradiance=irradiance,
#         temp_cell=real_temp,
#         alpha_sc=module['alpha_sc'],
#         a_ref=module['a_ref'],
#         I_L_ref=module['I_L_ref'],
#         I_o_ref=module['I_o_ref'],
#         R_sh_ref=module['R_sh_ref'],
#         R_s=module['R_s'],
#         EgRef=1.121,
#         dEgdT=-0.0002677
#     )
#     thermal_voltage = k*k_temp_real/q
#     expected = nNsVth/(Ns*thermal_voltage)

#     #adjust to per cell
#     result = physical_params.calc_ideality(k_temp_real, k_temp_ref, ref_n)
#     #Note: allow a slightly larger mismatch, due to a simpler model
#     assert np.isclose(result, expected, atol=0.1), f"Got {result}, expected {expected}"

#reference reverse saturation current
#ref_isat = module['I_o_ref']

@pytest.mark.parametrize("irradiance, real_temp", test_cases)
def test_isat(irradiance, real_temp):
    k_temp_real = real_temp+273.15
    k_temp_ref = ref_temp+273.15
    #get the expected values
    Iph, Isat, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
        effective_irradiance=irradiance,
        temp_cell=real_temp,
        alpha_sc=module['alpha_sc'],
        a_ref=module['a_ref'],
        I_L_ref=module['I_L_ref'],
        I_o_ref=module['I_o_ref'],
        R_sh_ref=module['R_sh_ref'],
        R_s=module['R_s'],
        EgRef=1.121,
        dEgdT=-0.0002677
    )

    expected = Isat
    result = physical_params.calc_isat(k_temp_real, k_temp_ref, ref_isat)
    print(f'Expected result is {Isat} actual result is {result}')
    #Note: allow a slightly larger mismatch, due to a simpler model
    assert np.isclose(result, expected, atol=0), f"Got {result}, expected {expected}"

#reference short circuit current, and iph
ref_alpha = module['alpha_sc']
ref_iph = module['I_L_ref']

@pytest.mark.parametrize("irradiance, real_temp", test_cases)
def test_iph(irradiance, real_temp):
    k_temp_real = real_temp+273.15
    k_temp_ref = ref_temp+273.15
    #get the expected values
    Iph, Isat, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
        effective_irradiance=irradiance,
        temp_cell=real_temp,
        alpha_sc=module['alpha_sc'],
        a_ref=module['a_ref'],
        I_L_ref=module['I_L_ref'],
        I_o_ref=module['I_o_ref'],
        R_sh_ref=module['R_sh_ref'],
        R_s=module['R_s'],
        EgRef=1.121,
        dEgdT=-0.0002677
    )

    expected = Iph
    result = physical_params.calc_iph(k_temp_real, k_temp_ref, ref_alpha, ref_iph, irradiance, ref_irradiance)
    #print(f'Expected Iph is {Iph} and result is {result}')
    #Note: allow a slightly larger mismatch, due to a simpler model
    assert np.isclose(result, expected, atol=0), f"Got {result}, expected {expected}"

@pytest.mark.parametrize("irradiance, real_temp", test_cases)
def test_Rsh(irradiance, real_temp):
    #get the expected values
    Iph, Isat, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
        effective_irradiance=irradiance,
        temp_cell=real_temp,
        alpha_sc=module['alpha_sc'],
        a_ref=module['a_ref'],
        I_L_ref=module['I_L_ref'],
        I_o_ref=module['I_o_ref'],
        R_sh_ref=module['R_sh_ref'],
        R_s=module['R_s'],
        EgRef=1.121,
        dEgdT=-0.0002677
    )

    expected = Rsh
    result = physical_params.calc_rsh(ref_rsh, irradiance, ref_irradiance)
    print(f'PVLib Rsh {expected} and actual is {result}')
    #Note: allow a slightly larger mismatch, due to a simpler model
    assert np.isclose(result, expected, atol=0), f"Got {result}, expected {expected}"
