"""
The purpose of this file is to calculate the reference conditions using known values should come from a datasheet
These use known points on the I-V curve to plot a curve 
Based off of https://journals.tubitak.gov.tr/cgi/viewcontent.cgi?article=1417&context=physics
"""

import numpy as np
import matplotlib.pyplot as plt

'''datasheet conditions: file:///C:/Users/Drew/OneDrive%20-%20University%20of%20Leeds/Documents/CHiplogic/Program/Jinko%20550M%20Solar%20Specs.pdf
isc = short circuit current
vmp = voltage at max power point
voc = open circuit voltage
imp = current at max power point
'''
isc = 14.01
vmp = 41.58
voc = 50.27
imp = 13.23

#get the approximations for c1 and c2
c1 = isc
c2 = (vmp - voc) / (np.log(1-imp/isc))

'''
@func the ITA function, can use with a linear solver to get a simulated I-V curve
@params - C1/C2 estimations - estimated constants
    - V - the voltage to generate
@returns ITA - the corresponding voltage
'''
def ITA(V, c1, c2):
    return isc - c1 * np.exp(-voc/c2) * (np.exp(V/c2)-1)

#generate simulated curve
volts = np.linspace(0, voc, 100)
currents = [ITA(V, c1, c2) for V in volts]

plt.figure(figsize=(8,5))
plt.plot(volts, currents, label='Synthetic I-V (single-diode)')
plt.plot([vmp, voc, 0], [imp, 0, isc], 'ro', label='Datasheet points')
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A]')
plt.title('Synthetic I-V curve using ITA-based extraction')
plt.grid(True)
plt.legend()

plt.savefig("synthetic_IV_curve.png", dpi=300)