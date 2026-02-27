from predicting_parameters import cell_ann
from predicting_parameters import refactored_prediction

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os

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
class Cell():
    #ANN at the class level only needs to be initiated once
    model = None
    X_scaler = None
    y_scaler = None

    #using rs as a constant
    rs = None

    #creating ann at class level
    @classmethod
    def initiate_class(cls, module_name, Ns):
        cls.model, cls.X_scaler, cls.y_scaler =  cell_ann.create_optimal_ann()

        #only need to get the rs everything else calculated by ann
        _, _, cls.rs, _, _, = refactored_prediction.getting_parameters(25, 1000, module_name)
        cls.rs = cls.rs/Ns
        

    #initiate with a irradiance, temperature and conditions from the datasheet
    #temp in kelvin
    def __init__(self, irr, temp, datasheet_conditions, module):
        self.irradiance = irr
        self.temperature = temp
        self.module_name = module

        self.kT = temp + 273.15
        self.Vth = k * self.kT / q

        self.isc, self.vmp, self.voc, self.imp, self.Ns = datasheet_conditions
        self.voc_per_cell = self.voc/self.Ns

        if Cell.model is None:
            Cell.initiate_class(self.module_name, self.Ns)

        self.predict_params()

    #using i-v law to calc voltage
    def iv_equation(self, V, I):
        exponent = (V + I * Cell.rs)/(self.a)
        exponent_term = self.isat * (np.exp(np.clip(exponent, -50, 50)) -1)
        rsh_term = (V + I * Cell.rs)/self.rsh
        return (self.iph - exponent_term - rsh_term - I)

    #use the ann
    def predict_params(self):
        X = [[self.irradiance, self.temperature]]
        X_scaled = Cell.X_scaler.transform(X)

        y_scaled = Cell.model.predict(X_scaled)
        y = Cell.y_scaler.inverse_transform(y_scaled)

        self.iph, log_isat, self.rsh, self.a = y[0]

        #reverse the logarithm
        self.isat = 10 ** log_isat

    #get the curve for graphing
    def get_curve(self):
        max_v = 0
        voltages = []

        #using solver to calculate V from I
        current_targets = np.linspace(0, self.iph, 200)
        v_guess = self.voc_per_cell * 0.99
        for I in current_targets:
            sol = fsolve(self.iv_equation, v_guess, args=(I,))[0]

            voltages.append(sol)
            v_guess = sol

        #output of the points
        return voltages, current_targets

    #set the irradiance and recalculate
    def shade(self, irr):
        self.irradiance = irr
        self.predict_params()

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
        test_cell = Cell(irr, temp, datasheet_conditions, 'Prism_Solar_Technologies_Bi48_267BSTC')
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

    filename = "cell_graphs/IV_comparison_ANN.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def cell_profiling(datasheet_conditions):
    irr_bounds = (100, 1000)
    temp_bounds = (25, 65)

    for i in range(50):
        irr = random.randint(*irr_bounds)
        temp = random.randint(*temp_bounds)

        test_cell = Cell(irr, temp, datasheet_conditions, 'Prism_Solar_Technologies_Bi48_267BSTC')
    #V, I = test_cell.get_curve()
    #     plt.plot(V, I)

    #     #plot with a small label
    #     idx = int(len(V) * 0.85)

    #     plt.text(
    #         V[idx],
    #         I[idx],
    #         f"G={irr}, T={temp}C",
    #         fontsize=1   # smaller text
    #     )

    # plt.xlabel("Voltage (V)")
    # plt.ylabel("Current (A)")
    # plt.title("I–V Curves Under Different Conditions")
    # plt.grid(True)

    # filename = "cell_graphs/Many_IV_comparison_ANN.png"
    # plt.savefig(filename, dpi=300, bbox_inches="tight")
    # plt.show()
    # plt.close()


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
    test_cases = [
        (100, 25),
        (500, 25),
        (500, 35),
        (1000, 25),
        (1000, 45),
        (300, 25),
        (900, 65),
    ]

    #plot_test_cases(test_cases)

    run_profile()