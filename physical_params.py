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

#frequently used variables
#boltzmans constant
k = 1.38 * (10 ** -23)
#electron charge
q = 1.6 * (10 ** -19)
#constant for band gap energy
Eg = 1.21 * (10 ** -19)
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
def calc_ideality(real_temp, ref_temp, ref_a):
    return ref_a/(k*real_temp/q)

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
    exponent = Eg/k * ((1/ref_temp) - (1/real_temp))
    return ref_isat * (real_temp/ref_temp)**3 * np.exp(exponent)

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