import pvlib
import pytest
import numpy as np
import physical_params

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
ref_a = module['a_ref']/Ns

#in form (irradiance, real temperature)
test_cases = [
    (1000, 25),
    (1000, 45),
    (500, 25),
    (500, 45),
    (100, 25),
    (100, 45)
]

@pytest.mark.parametrize("irradiance, real_temp", test_cases)
def test_ideality(irradiance, real_temp):
    k_temp_real = real_temp+273.15
    k_temp_ref = ref_temp+273.15
    #get the expected values
    Iph, Isat, nNsVth, Rs, Rsh = pvlib.pvsystem.calcparams_cec(
        effective_irradiance=irradiance,
        temp_cell=real_temp,
        alpha_sc=module['alpha_sc'],
        a_ref=ref_a,
        I_L_ref=module['I_L_ref'],
        I_o_ref=module['I_o_ref'],
        R_sh_ref=module['R_sh_ref'],
        R_s=module['R_s'],
        Adjust=module['Adjust']
    )
    thermal_voltage = k*k_temp_real/q
    expected = nNsVth/(Ns*thermal_voltage)

    #adjust to per cell
    a_ref_cell = ref_a/Ns
    result = physical_params.calc_ideality(k_temp_real, k_temp_ref, a_ref_cell)
    #Note: allow a slightly larger mismatch, due to a simpler model
    assert np.isclose(result, expected, atol=0.2), f"Got {result}, expected {expected}"

#reference reverse saturation current
ref_isat = module['I_o_ref']

@pytest.mark.parametrize("irradiance, real_temp", test_cases)
def test_isat(irradiance, real_temp):
    k_temp_real = real_temp+273.15
    k_temp_ref = ref_temp+273.15
    #get the expected values
    Iph, Isat, nNsVth, Rs, Rsh = pvlib.pvsystem.calcparams_cec(
        effective_irradiance=irradiance,
        temp_cell=real_temp,
        alpha_sc=module['alpha_sc'],
        a_ref=ref_a,
        I_L_ref=module['I_L_ref'],
        I_o_ref=module['I_o_ref'],
        R_sh_ref=module['R_sh_ref'],
        R_s=module['R_s'],
        Adjust=module['Adjust']
    )

    expected = Isat
    result = physical_params.calc_isat(k_temp_real, k_temp_ref, ref_isat)
    #Note: allow a slightly larger mismatch, due to a simpler model
    assert np.isclose(result, expected, atol=1e-2), f"Got {result}, expected {expected}"

#reference short circuit current, and iph
ref_alpha = module['alpha_sc']
ref_iph = module['I_L_ref']

@pytest.mark.parametrize("irradiance, real_temp", test_cases)
def test_isat(irradiance, real_temp):
    #get the expected values
    Iph, Isat, nNsVth, Rs, Rsh = pvlib.pvsystem.calcparams_cec(
        effective_irradiance=irradiance,
        temp_cell=real_temp,
        alpha_sc=module['alpha_sc'],
        a_ref=ref_a,
        I_L_ref=module['I_L_ref'],
        I_o_ref=module['I_o_ref'],
        R_sh_ref=module['R_sh_ref'],
        R_s=module['R_s'],
        Adjust=module['Adjust']
    )

    expected = Iph
    result = physical_params.calc_iph(real_temp, ref_temp, ref_alpha, ref_iph, irradiance, ref_irradiance)
    #Note: allow a slightly larger mismatch, due to a simpler model
    assert np.isclose(result, expected, atol=1e-2), f"Got {result}, expected {expected}"

