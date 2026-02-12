import reference_conditions
import physical_params
import numpy as np
from scipy.optimize import fsolve, least_squares
import os
import matplotlib.pyplot as plt

import pvlib
cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
module = cec_modules['Canadian_Solar_Inc__CS1U_405MS']

#reference boltzmans and electrical charge
k = 1.38e-23
q = 1.6e-19

#a class used to calculate the parameters of a single cell
class cell():
    #initiate with a irradiance, temperature and conditions from the datasheet
    #temp in kelvin
    def __init__(self, irr, temp, datasheet_conditions, alpha_isc):
        self.irr = irr
        self.kT = temp
        self.isc, self.vmp, self.voc, self.isc, self.Ns = datasheet_conditions
        self.alpha_isc = alpha_isc
        self.Vth = k * self.kT / q

        #calculate the reference params
        reference_params = reference_conditions.get_reference_params((self.isc, self.vmp, self.voc, self.isc, self.Ns))

        #then calculate the physical conditions
        actual_params = physical_params.return_adjusted(reference_params, (irr, temp), self.alpha_isc)
        self.iph, self.isat, self.rs, self.rsh, self.n = actual_params
        self.nNsVth = self.n * self.Ns * self.Vth

        #run all the methods to get the actual params given the conditions
        self.get_resist()

    #using i-v law to calc voltage
    def iv_equation(self, V, I):
        exponent = (V + I * self.rs)/(self.nNsVth)
        exponent_term = self.isat * (np.exp(np.clip(exponent, -50, 50)) -1)
        rsh_term = (V + I * self.rs)/self.rsh
        return (self.iph - exponent_term - rsh_term - I)

    #generate points on an I-V curve
    def get_points(self):
        max_v = 0
        voltages = []

        #using solver to calculate V from I
        current_targets = np.linspace(0, self.iph, 30)
        for I in current_targets:
            v_guess = 0.7
            sol = fsolve(self.iv_equation, v_guess, args=(I,))[0]

            voltages.append(sol)

        #output of the points
        output = [(V, I) for V, I in zip(voltages, current_targets)]
        return output

    #use a solver to get rs and rsh
    def get_resist(self):
        points = self.get_points()
        #set nsvth to new value with Ns
        self.nNsVth = self.n * self.Vth

        def residuals(x):
            Rs, Rsh = x
            res = []
            for V, I in points:
                I_calc = self.iph - self.isat*(np.exp(np.clip((V + I*Rs)/(self.nNsVth), -50, 50)) - 1) - (V + I*Rs)/Rsh
                res.append(I_calc - I)
            return res

        # initial guess
        x0 = [0.1, 1000]
        sol = least_squares(residuals, x0, bounds=([0, 0],[5, 1e6]))
        self.rs, self.rsh = sol.x

    #draw a final graph
    def plot_cell_graph(self):
        points = self.get_points()
        V = [p[0] for p in points]
        I = [p[1] for p in points]

        plt.figure()
        plt.plot(V, I)
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title(f"I–V Curve | G={self.irr} W/m², T={self.kT} K")
        plt.grid(True)

        # save
        filename = f"cell_graphs/IV_G{self.irr}_T{int(self.kT)}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    #get the curve for graphing
    def get_curve(self):
        points = self.get_points()
        V = [p[0] for p in points]
        I = [p[1] for p in points]
        return V, I

#use the test conditions to generate curves for each conditions
def plot_test_cases(test_cases):
    os.makedirs("test_graphs", exist_ok=True)
    alpha_isc = 0.05
    params = (9.65, 44.3, 53.5, 9.16, 81)

    plt.figure()

    for irr, temp in test_cases:
        test_cell = cell(irr, temp, params, alpha_isc)
        V, I = test_cell.get_curve()

        #remove v < 0
        V_filtered = [v for v in V if v >= 0]
        I_filtered = [I[i] for i, v in enumerate(V) if v >= 0]

        plt.plot(V_filtered, I_filtered)

        #plot with a small label
        idx = int(len(V_filtered) * 0.85)

        plt.text(
            V_filtered[idx],
            I_filtered[idx],
            f"G={irr}, T={temp}K",
            fontsize=6   # smaller text
        )

    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("I–V Curves Under Different Conditions")
    plt.grid(True)

    filename = "cell_graphs/IV_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

#need temp to be in kelvin
test_cases = [
    (100, 298.15),
    (500, 298.15),
    (500, 318.15),
    (1000, 298.15),
    (1000, 318.15),
    (300, 308.15),
    (900, 333.15),
]

plot_test_cases(test_cases)