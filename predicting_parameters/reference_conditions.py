"""
The purpose of this file is to calculate the reference conditions using known values should come from a datasheet
These use known points on the I-V curve to plot a curve 
Based off of https://journals.tubitak.gov.tr/cgi/viewcontent.cgi?article=1417&context=physics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, fsolve
import pvlib 
from pvlib.pvsystem import i_from_v
import os

'''datasheet conditions: https://cdn.enfsolar.com/Product/pdf/Crystalline/5bbdb7c34b4dc.pdf
isc = short circuit current
vmp = voltage at max power point
voc = open circuit voltage
imp = current at max power point
'''
isc = 9.65
vmp = 44.3
voc = 53.5
imp = 9.16
Ns = 81

# #get the approximations for c1 and c2
# c1 = isc
# c2 = (vmp - voc) / (np.log(1-imp/isc))

#using least square to solve c1 and c2
def c_approx(c1, c2):
    x0 = [c1, c2]

    def c_approx_residuals(x):
        c1, c2 = x
        res = []

        #equation 5
        voc_exp = np.exp(-voc/c2)
        res.append(c1 - isc/(1-voc_exp))

        #equation 6
        numer = isc - imp
        vmp_exp = np.exp(vmp/c2)
        denom = (vmp_exp-1) * voc_exp
        res.append(c1 - numer/denom)

        return np.array(res)

    sol = least_squares(c_approx_residuals, x0, args=())
    return sol.x


'''
@func the ITA function, can use with a linear solver to get a simulated I-V curve
@params - C1/C2 estimations - estimated constants
    - V - the voltage to generate
@returns ITA - the corresponding voltage
'''
def ITA(V, c1, c2):
    voc_exp = np.exp(np.clip(-voc/c2, -50, 50))
    v_exp = np.exp(np.clip(V/c2, -50, 50))
    return isc - c1 * voc_exp * (v_exp-1)

#solving simultaneous to get correct c1/c2
# new_c1, new_c2 = c_approx(c1, c2)

# #generating curve
# volts = np.linspace(0, voc, 300)
# ita_currents = [ITA(V, new_c1, new_c2) for V in volts]

# #comparing to pvlib
# cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
# module = cec_modules['Canadian_Solar_Inc__CS1U_405MS']
# NsVth = k*298.15*Ns/q

# Iph, I0, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
#     effective_irradiance=1000,
#     temp_cell=25,
#     alpha_sc=module['alpha_sc'],
#     a_ref=module['a_ref'],
#     I_L_ref=module['I_L_ref'],
#     I_o_ref=module['I_o_ref'],
#     R_sh_ref=module['R_sh_ref'],
#     R_s=module['R_s'],
#     EgRef=1.121,
#     dEgdT=-0.0002677
# )

# lib_currents = i_from_v(volts, Iph, I0, Rs, Rsh, nNsVth)

# #Create folder if it doesn't exist
# os.makedirs("graphs", exist_ok=True)

# # Plot both curves
# plt.figure(figsize=(8,5))
# plt.plot(volts, ita_currents, label='ITA Model', color='blue')
# plt.plot(volts, lib_currents, label='pvlib Model', color='orange', linestyle='--')
# plt.xlabel("Voltage [V]")
# plt.ylabel("Current [A]")
# plt.title("I-V Curve Comparison")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# # Save the figure
# plt.savefig("graphs/IV_comparison.png", dpi=300)
# plt.show()


#use least square to find the correct reference conditions
def desoto_residuals(x, Isat, isc, vmp, voc, imp, Ns, residuals_arr=[], values_history=[]):
    #residuals to minimuse 
    res = []
    
    k = 1.38e-23
    q = 1.6e-19

    #get the parameters from x
    Iph, Rs, Rsh, n = x
    vth = (n*k*298.15*Ns)/q

    exponent = isc*Rs/vth
    exp_term = Isat * (np.exp(exponent) - 1)
    shunt_term = (isc*Rs)/Rsh
    output = Iph - exp_term - shunt_term - isc
    res.append(output)

    exponent = voc/vth
    exp_term = Isat * (np.exp(exponent) - 1)
    shunt_term = voc/Rsh
    output = Iph - exp_term - shunt_term
    res.append(output)

    exponent = (vmp+imp*Rs)/vth
    exp_term = Isat * (np.exp(exponent) - 1)
    shunt_term = (vmp + imp*Rs)/Rsh
    output = Iph - exp_term - shunt_term - imp
    res.append(output)

    exponent = (vmp+imp*Rs)/vth
    exponent = np.exp(exponent)
    numerator = (-Isat/vth)*exponent-1/Rsh
    denom = 1+((Isat*Rs)/vth)*exponent+Rs/Rsh
    output = imp + vmp*(numerator/denom)
    res.append(output)

    residuals_arr.append(np.linalg.norm(res))
    values_history.append(x)
    return np.array(res)

#using the ITA method of parameter extraction
def ita_residuals(x, currents, volts, Isat):
    #residuals to minimuse
    res = []

    #get the parameters from x
    Iph, Rs, Rsh, n = x
 
    vth = (n*k*298.15*Ns)/q
    for j, V in enumerate(volts):
        I = currents[j]

        #I-V calculation at reference conditions (for 1 cell)
        exponent = (V+I*Rs)/vth
        exp_term = Isat * (np.exp(np.clip(exponent, -700, 700)))
        shunt_term = (V + I*Rs)/Rsh

        residual = -I + Iph - exp_term - shunt_term
        res.append(residual)

    #testing equation 14 - iph rs and rsh 
    # output = Iph - (Rsh+Rs)/Rsh * isc
    # res.append(output*10)

    #testing gradient residuals
    slope_sc = (currents[1] - currents[0]) / (volts[1] - volts[0])
    slope_calc_sc = -Isat/vth - 1/Rsh
    output = slope_calc_sc - slope_sc
    res.append(output*10)

    numerator = vmp*(vmp+imp*Rs)
    exponent = np.exp(np.clip((vmp+imp*Rs)/vth, -700, 700))
    denom = (vmp*Iph) - (vmp*Isat*exponent) + (vmp*Isat) - (vmp*imp)
    res.append(Rsh - numerator/denom)

    return np.array(res)

##compare how smooth the iv curve is for both 
def compare_methods():

    Isat = 2.809e-11

    x0 = [isc, 0.1, 100, 1]
    bounds = (
        [0.9*isc, 0.1, 100, 0.5],
        [1.1*isc, 2, 1000, 2.0]
    )   

    ita_sol = least_squares(ita_residuals, x0, args=(volts, lib_currents, Isat,), bounds=bounds)
    desoto_sol = least_squares(desoto_residuals, x0, args=(Isat,), bounds=bounds)

    ita_voltages = generate_curve(lib_currents, ita_sol.x, Isat)
    desoto_voltages = generate_curve(lib_currents, desoto_sol.x, Isat)

    print(f" iph = {desoto_sol.x[0]:.6e} A")
    #print(f" isat = {ita_sol.x[1]:.6e} A")
    print(f" rs = {desoto_sol.x[1]:.6f} Ω")
    print(f" rsh = {desoto_sol.x[2]:.6f} Ω")
    print(f" n = {desoto_sol.x[3]:.4f}")
        
    plt.figure(figsize=(8,5))
    plt.plot(ita_voltages, lib_currents, label='ITA Model', color='blue')
    plt.plot(desoto_voltages, lib_currents, label='DeSoto Fit', color='orange', linestyle='--')
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A]")
    plt.title("I-V Curve Comparison: ITA vs DeSoto")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig("graphs/IV_comparison_methods.png", dpi=300)
    plt.show()

def generate_curve(currents, x, Isat):
    Iph, Rs, Rsh, n = x
    v_guess = .3
    Ns = 81

    def iv_equation(V, I):
        exponent = q*(V + I * Rs)/(n*Ns*k*298.15)
        exponent_term = Isat * (np.exp(np.clip(exponent, -50, 50)) -1)
        rsh_term = (V + I * Rs)/Rsh
        return (Iph - exponent_term - rsh_term - I)

    voltages = [fsolve(iv_equation, v_guess, args=(I,))[0] for I in currents]
    return voltages

'''
@func uses the ITA to generate points on a curve, then a least square solver to get the paramters
---- CURRENTLY USES THESE FROM THE FILE WILL ADJUST FOR LATER WORK ----
@params - needs c1/c2 estimations,
    - the datasheet conditions
@returns sol.x - the parameters of the I-V equation at reference condition
'''
def get_reference_params(params, verbose=False):
    #datasheet conditions
    isc, vmp, voc, imp, Ns = params
    #reference boltzmans and electrical charge
    k = 1.38e-23
    q = 1.6e-19

    #Iph, Isat, Rs, Rsh, n - initial guess    
    x0 = [isc, 0.1, 100, 1]

    #Iph, Isat, Rs, Rsh, n   
    #keeping current and resistance positive, and n between 0.5 and 2
    bounds = (
        [0.9*isc, 0.1, 100, 0.5],
        [1.1*isc, 2, 1000, 2.0]
    )
    residuals_arr = []
    values_history = []

    Isat = 2.809e-11

    sol = least_squares(desoto_residuals, x0, args=(2.809e-11, isc, vmp, voc, imp, Ns, residuals_arr, values_history,), bounds=bounds)

    if verbose:
        print(f" iph = {sol.x[0]:.6e} A")
        print(f" isat = {Isat:.6e} A")
        print(f" rs = {sol.x[1]:.6f} Ω")
        print(f" rsh = {sol.x[2]:.6f} Ω")
        print(f" n = {sol.x[3]:.4f}")

        #graphing residuals against results
        iters = np.arange(len(residuals_arr))

        plt.figure(figsize=(14, 7))   # BIGGER figure
        plt.semilogy(iters, residuals_arr, marker='o', markersize=3)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("‖Residual‖", fontsize=12)
        plt.title("Residual convergence", fontsize=14)
        plt.grid(True, alpha=0.4)

        # annotate every Nth point (fewer labels)
        step = max(1, len(iters) // 12)

        for i in range(0, len(iters), step):
            Iph, Rs, Rsh, n = values_history[i]
            label = f"Iph={Iph:.2f}, Rs={Rs:.2f}, Rsh={Rsh:.0f}, n={n:.2f}"
            plt.annotate(
                label,
                (iters[i], residuals_arr[i]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,        # SMALLER text
                alpha=0.8,
                rotation=45
            )

        plt.tight_layout()
        plt.savefig(
            "residual_vs_iteration_annotated.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


    return sol.x[0], 2.809e-11, sol.x[1], sol.x[2], sol.x[3]

a_guess = 0.027*Ns - 0.0172