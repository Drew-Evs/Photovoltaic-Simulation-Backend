from predicting_parameters import reference_conditions
from predicting_parameters import physical_params
from predicting_parameters import refactored_prediction
import numpy as np
from scipy.optimize import fsolve, least_squares
import os
import matplotlib.pyplot as plt

#for profiling
import cProfile
import pstats
import io
import random

import pvlib
cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
module = cec_modules['Prism_Solar_Technologies_Bi48_267BSTC']
module_name = 'Prism_Solar_Technologies_Bi48_267BSTC'

#reference boltzmans and electrical charge
k = 1.38e-23
q = 1.6e-19

#a class used to calculate the parameters of a single cell
class cell():
    #initiate with a irradiance, temperature and conditions from the datasheet
    #temp in kelvin
    def __init__(self, irr, temp, datasheet_conditions, module_name):
        self.irr = irr
        self.kT = temp + 273.15
        self.isc, self.vmp, self.voc, self.imp, self.Ns = datasheet_conditions
        self.voc_per_cell = self.voc/self.Ns
        self.Vth = k * self.kT / q

        #using refactored params to calculate
        actual_params = refactored_prediction.getting_parameters(temp, irr, module_name)

        #refactor to use self.a (modified ideality factor)
        self.iph, self.isat, self.rs, self.rsh, self.a = actual_params

        #run all the methods to get the actual params given the conditions
        self.get_resist()


    #using i-v law to calc voltage
    def iv_equation(self, V, I):
        exponent = (V + I * self.rs)/(self.a)
        exponent_term = self.isat * (np.exp(np.clip(exponent, -50, 50)) -1)
        rsh_term = (V + I * self.rs)/self.rsh
        return (self.iph - exponent_term - rsh_term - I)

    #generate points on an I-V curve
    def get_points(self):
        max_v = 0
        voltages = []

        #using solver to calculate V from I
        current_targets = np.linspace(0, self.iph, 200)
        v_guess = self.voc_per_cell*self.Ns
        for I in current_targets:
            sol = fsolve(self.iv_equation, v_guess, args=(I,))[0]

            voltages.append(sol)
            v_guess = sol

        #output of the points
        output = [(V, I) for V, I in zip(voltages, current_targets)]
        return output

    #use a solver to get rs and rsh
    def get_resist(self):
        points = self.get_points()
        points = [(V/self.Ns, I) for (V, I) in points]
        #set nsvth to new value per cell
        self.a = self.a / self.Ns

        def residuals(x):
            Rs, Rsh = x
            res = []
            for V, I in points:
                I_calc = self.iph - self.isat*(np.exp(np.clip((V + I*Rs)/(self.a), -50, 50)) - 1) - (V + I*Rs)/Rsh
                res.append(I_calc - I)
            return res

        # initial guess
        x0 = [self.rs/self.Ns, self.rsh/self.Ns]
        sol = least_squares(residuals, x0, bounds=([0.001, 1],[75, 1e6]))
        self.Ns = 1
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
    datasheet_conditions = (
        module['I_sc_ref'], 
        module['V_mp_ref'], 
        module['V_oc_ref'], 
        module['I_mp_ref'],
        module['N_s']
    )

    plt.figure()

    for irr, temp in test_cases:
        test_cell = cell(irr, temp, datasheet_conditions, 'Prism_Solar_Technologies_Bi48_267BSTC')
        V, I = test_cell.get_curve()

        plt.plot(V, I)

        #plot with a small label
        idx = int(len(V) * 0.85)

        plt.text(
            V[idx],
            I[idx],
            f"G={irr}, T={temp}C",
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


def cell_profiling(datasheet_conditions):
    irr_bounds = (100, 1000)
    temp_bounds = (25, 65)

    for i in range(50):
        irr = random.randint(*irr_bounds)
        temp = random.randint(*temp_bounds)

        test_cell = cell(irr, temp, datasheet_conditions, 'Prism_Solar_Technologies_Bi48_267BSTC')


def run_profile():
    datasheet_conditions = (
        module['I_sc_ref'], 
        module['V_mp_ref'], 
        module['V_oc_ref'], 
        module['I_mp_ref'],
        module['N_s']
    )
    pr = cProfile.Profile()
    pr.enable()

    cell_profiling(datasheet_conditions)

    pr.disable()

    #sort values
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)

    print(s.getvalue())

if __name__ == "__main__":
    #need temp to be in kelvin
    test_cases = [
        (100, 25),
        (500, 25),
        (500, 35),
        (1000, 25),
        (1000, 45),
        (300, 25),
        (900, 65),
    ]

    plot_test_cases(test_cases)


'''
commands to run
python -m cProfile -o output.prof single_cell.py

then view with snakeviz
pip install snakeviz
snakeviz output.prof
'''