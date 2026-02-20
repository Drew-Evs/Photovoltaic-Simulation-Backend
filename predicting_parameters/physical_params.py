"""
The purpose of this file is to calculate the phyiscal parameters used in Kirchoffs I-V equation
These include: - n - ideality factor
    - isat - reverse saturation current
    - iph - photocurrent
    - Rs - series resistance
    - Rsh - parallel (aka shunt) resistance
Based off of https://www.sciencedirect.com/science/article/pii/S0038092X05002410?via%3Dihub#aep-section-id20
Reference conditions normally taken from a manufacturer specification sheet
These include: - αIsc - temperature coefficient for short circuit current
    - STC - standard operating conditions (irradiance/temperature/airmass)
    - NOCT - nominal operating cell temperature
"""
import numpy as np
import pvlib

cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
module = cec_modules['Canadian_Solar_Inc__CS1U_405MS']

#frequently used variables
#boltzmans constant
k = 1.38 * (10 ** -23)
k_ev = 8.617333e-5 #same variable in ev for use with band gap energy
#electron charge
q = 1.6 * (10 ** -19)
#constant for band gap energy
Eg_ref = 1.121
#using constants for air mass and reference air mass
#IMPORTANT - look this up later see how to develop it
m = 1.5
m_ref = 1.5


"""
@func calculating the diode ideality factor
    this is simply relative to temperature
@params - operating temperature of the module
    - reference temperature
    - reference ideality factor
@output operating ideality factor
"""
def calc_ideality(real_temp, ref_temp, ref_n):
    return ref_n*(real_temp/ref_temp)

"""
@func calculating reverse saturation current
@params - operating temperature of the module
    - reference temperature
    - reference reverse saturation
    - band gap energy (Eg- needs to be calculated/found beforehand) at temp
    IMPORTANT - for now use 1.21*(10**-19) - research calculation later (wavelength a solar cell can absorb)
@output operating reverse saturation current
"""
def calc_isat(real_temp, ref_temp, ref_isat):
    real_eg = calc_band_gap(ref_temp, real_temp)
    exponent = (1/k_ev) * (Eg_ref/ref_temp - real_eg/real_temp)
    return ref_isat * (real_temp/ref_temp)**3 * np.exp(exponent)    
'''IMPORTANT - needs changing formula incorrect'''

def calc_band_gap(T, T_ref=298.15):
    return Eg_ref * (1 - 0.0002677 * (T - T_ref))

"""
@func calculating photocurrent generated
    this is based on the temperature of the cell
    and the amoutn of irradiance that gets through
@params - operating temperature of the module
    - reference temperature
    - operating airmass of the module
    - reference airmass
    - reference temperature coefficient for short circuit current
    - reference photocurrent at STC
    - operating irradiacne of the module
    - reference irradiance at STC
@output operating photocurrent
"""
def calc_iph(real_temp, ref_temp, alpha_isc, ref_iph, real_g, ref_g):
    return (real_g/ref_g) * (m/m_ref) * (ref_iph + alpha_isc * (real_temp - ref_temp)) 

"""
@func calculating shunt resistance as a function of actual irradiance
@params - reference shunt resistance
    - operating irradiacne of the module
    - reference irradiance at STC
@output operating shunt resistance
"""
def calc_rsh(ref_rsh, real_g, ref_g):
    return ref_rsh * (ref_g/real_g)

'''
Final function to take in the initial and then output the adjusted parameters 
for irradiance and temperature

@params - ref_params : tuple
            (ref_iph, ref_isat, ref_rs, ref_rsh, ref_n)
        - actual_conditions : tuple
            (real irradiance, real temperature)
        - alphas_isc : float
            the temp coefficient of Isc

@output - real_iph : the actual photocurrent 
        - real_isat : the operating saturation current
        - real_rsh : operating shunt current
        - real_n : operaitng ideality factor
'''
def return_adjusted(ref_params, actual_conditions, alpha_isc):
    ref_irr, ref_temp = 1000, 298.15
    ref_iph, ref_isat, ref_rs, ref_rsh, ref_n = ref_params
    real_irr, real_temp = actual_conditions

    # #print(f'Getting real irr as {real_irr}')

    # #frequently used variables
    k = 1.38e-23
    q = 1.6e-19

    Vth = k * actual_conditions[1] / q

    #constant for band gap energy
    Eg = 1.21 * (10 ** -19)
    #using constants for air mass and reference air mass
    m = 1.5
    m_ref = 1.5

    # ''' ERROR IS SOMEHOW HERE WITH THE RSH IMPORTANT !!!!!'''
    real_isat = calc_isat(real_temp, ref_temp, ref_isat)
    real_iph = calc_iph(real_temp, ref_temp, alpha_isc, ref_iph, real_irr, ref_irr)
    real_rsh = calc_rsh(ref_rsh, real_irr, ref_irr)
    # #print(f'Initial Rsh {ref_rsh} and actual is {real_rsh}')

    Iph, Is, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
        effective_irradiance=actual_conditions[0],
        temp_cell=(actual_conditions[1]-273.15),
        alpha_sc=module['alpha_sc'],
        a_ref=module['a_ref'],
        I_L_ref=module['I_L_ref'],
        I_o_ref=module['I_o_ref'],
        R_sh_ref=module['R_sh_ref'],
        R_s=module['R_s'],
        EgRef=1.121,
        dEgdT=-0.0002677
    )
    n = nNsVth/(module['N_s']*Vth)

    # print(f'Before adjustment')
    # print(f"Iph={Iph} Isat={Is}, Rs={Rs}, Rsh={Rsh}, n={n}")

    #return Iph, Is, Rs, Rsh, n


    #from testing use the reference n
    #need to fix to use generate the IS better
    return real_iph, Is, ref_rs, real_rsh, ref_n
