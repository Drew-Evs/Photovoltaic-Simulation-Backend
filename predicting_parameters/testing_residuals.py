'''this page is composed of the functions used to test competency of residuals'''
'''using the desoto model - testing if better'''
def test_desoto_model():
    #testing equation 3 for residual of iph
    Iph = 9.6532
    Isat = 2.809e-11 
    Rs = 0.3288
    Rsh = 978.24
    n = 0.968
    vth = (n*k*298.15*Ns)/q

    exponent = isc*Rs/vth
    exp_term = Isat * (np.exp(exponent) - 1)
    shunt_term = (isc*Rs)/Rsh
    output = Iph - exp_term - shunt_term - isc
    print(f'Residual of short circuit current formula {output}')

    exponent = voc/vth
    exp_term = Isat * (np.exp(exponent) - 1)
    shunt_term = voc/Rsh
    output = Iph - exp_term - shunt_term
    print(f'Residual of open circuit voltage formula {output}')

    exponent = (vmp+imp*Rs)/vth
    exp_term = Isat * (np.exp(exponent) - 1)
    shunt_term = (vmp + imp*Rs)/Rsh
    output = Iph - exp_term - shunt_term - imp
    print(f'Residual of max power point formula {output}')

    exponent = (vmp+imp*Rs)/vth
    exponent = np.exp(exponent)
    numerator = (-Isat/vth)*exponent-1/Rsh
    denom = 1+((Isat*Rs)/vth)*exponent+Rs/Rsh
    output = imp + vmp*(numerator/denom)
    print(f'Residual of max power point derivative {output}')


#trying to test the output of varying Rsh on gradient residual
def graph_gradient_residual():
    volts = np.linspace(0, 10, 10)
    currents = [ITA(V, c1, c2) for V in volts]

    #using almost correct conditions to test residuals
    Iph = isc
    Isat = 1e-12
    Rs = 0.32
    Rsh_arr = np.linspace(0, 1200, 25)
    n = 1
    vth = (n*k*298.15*Ns)/q

    residuals = []
    for Rsh in Rsh_arr:
        #testing gradient residuals
        slope_sc = (currents[1] - currents[0]) / (volts[1] - volts[0])
        slope_oc = (currents[-1] - currents[-2]) / (volts[-1] - volts[-2]) 
        #short circuit
        slope_calc_sc = -Isat/vth - 1/Rsh
        residuals.append(slope_calc_sc - slope_sc)

    plt.figure()
    plt.plot(Rsh_arr, residuals)
    plt.xlabel("Parallel resistance Rsh (Ω)")
    plt.ylabel("Total residual")
    plt.title("Residual vs Parallel Resistance")
    plt.grid(True)

    plt.savefig("gradient_residual_vs_Rsh.png", dpi=300, bbox_inches="tight")
    plt.close()


#trying to test the output of varying RS on I-V residual
def graph_i_v_residual():
    volts = np.linspace(0, 10, 10)
    currents = [ITA(V, c1, c2) for V in volts]

    #using almost correct conditions to test residuals
    Iph = isc
    Isat = 1e-12
    Rs_arr = np.linspace(0, 1, 25)
    Rsh = 950
    n = 1
    vth = (n*k*298.15*Ns)/q

    residuals = []
    for Rs in Rs_arr:
        residual = 0
        for j, V in enumerate(volts):
            I = currents[j]

            #I-V calculation at reference conditions (for 1 cell)
            exponent = (V+I*Rs)/vth
            exp_term = Isat * (np.exp(np.clip(exponent, -700, 700))-1)
            shunt_term = (V + I*Rs)/Rsh

            residual += -I + Iph - exp_term - shunt_term
        residuals.append(residual/10)

    plt.figure()
    plt.plot(Rs_arr, residuals)
    plt.xlabel("Series resistance Rs (Ω)")
    plt.ylabel("Total residual")
    plt.title("Residual vs Series Resistance")
    plt.grid(True)

    plt.savefig("i_v_residual_vs_Rs.png", dpi=300, bbox_inches="tight")
    plt.close()

def output_residual():
    volts = np.linspace(0, voc, 10)
    currents = [ITA(V, c1, c2) for V in volts]

    #using almost correct conditions to test residuals
    Iph = isc
    Isat = 1e-12
    Rs = 2
    Rsh = 950
    n = 1
    vth = (n*k*298.15*Ns)/q

    output = 0
    for j, V in enumerate(volts):
        I = currents[j]

        #I-V calculation at reference conditions (for 1 cell)
        exponent = (V+I*Rs)/vth
        exp_term = Isat * (np.exp(np.clip(exponent, -700, 700))-1)
        shunt_term = (V + I*Rs)/Rsh

        output += -I + Iph - exp_term - shunt_term
    print(f'Average I-V residual is {output/10}')

    #testing residual of the Rsh formula
    numerator = vmp*(vmp+imp*Rs)
    exponent = np.exp(np.clip((vmp+imp*Rs)/vth, -700, 700))
    denom = (vmp*Iph) - (vmp*Isat*exponent) + (vmp*Isat) - (vmp*imp)
    output = Rsh - numerator/denom
    print(f'Residual of Rsh formula is {output}')

    #testing equation 14 - iph rs and rsh 
    output = Iph - (Rsh+Rs)/Rsh * isc
    print(f'Residual of Iph, Rs and Rsh formula is {output}')

    #testing gradient residuals
    slope_sc = (currents[1] - currents[0]) / (volts[1] - volts[0])
    slope_oc = (currents[-1] - currents[-2]) / (volts[-1] - volts[-2]) 
    #short circuit
    slope_calc_sc = -Isat/vth - 1/Rsh
    ouput = slope_calc_sc - slope_sc
    print(f'Residual of short circuit gradiaent formula is {output}')

    # VOC slope residual
    slope_calc_oc = -Isat/vth * np.exp(voc/vth) - 1/Rsh
    output = slope_calc_oc - slope_oc
    print(f'Residual of open circuit gradiaent formula is {output}')