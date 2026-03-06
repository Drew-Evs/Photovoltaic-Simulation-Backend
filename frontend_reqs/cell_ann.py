from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import os
import pvlib
import numpy as np

from scipy.optimize import fsolve, least_squares

import refactored_prediction

#used to build the database if needed
class DataEntry():
    def __init__(self, irr, temp, datasheet_conditions, module_name):
        #datasheet conditions and parameter modelling
        self.isc, self.vmp, self.voc, self.imp, self.Ns = datasheet_conditions
        actual_params = refactored_prediction.getting_parameters(temp, irr, module_name)
        self.iph, self.isat, self.rs, self.rsh, self.a = actual_params
        self.get_resist()

    #IV equation   
    def iv_equation(self, V, I):
        exponent = (V + I * self.rs)/(self.a)
        exponent_term = self.isat * (np.exp(np.clip(exponent, -50, 50)) -1)
        rsh_term = (V + I * self.rs)/self.rsh
        return (self.iph - exponent_term - rsh_term - I)

    #generate IV points for resistance
    def get_points(self):
        voltages = []
        current_targets = np.linspace(0, self.iph, 200)
        v_guess = self.voc_per_cell*self.Ns
        for I in current_targets:
            sol = fsolve(self.iv_equation, v_guess, args=(I,))[0]
            voltages.append(sol)
        output = [(V, I) for V, I in zip(voltages, current_targets)]
        return output

    #get individual cell resistance 
    def get_resist(self):
        points = self.get_points()
        points = [(V/self.Ns, I) for (V, I) in points]
        self.a = self.a / self.Ns
        def residuals(x):
            Rs, Rsh = x
            res = []
            for V, I in points:
                I_calc = self.iph - self.isat*(np.exp(np.clip((V + I*Rs)/(self.a), -50, 50)) - 1) - (V + I*Rs)/Rsh
                res.append(I_calc - I)
            return res
        x0 = [self.rs/self.Ns, self.rsh/self.Ns]
        sol = least_squares(residuals, x0, bounds=([0.001, 1],[75, 1e6]))
        self.Ns = 1
        self.rs, self.rsh = sol.x

def create_dataset(module_name):
    #generate randomly a dataset using the cell calculation
    x0 = []
    y0 = []

    #sees if data exists and if not generates own 
    if os.path.exists(f"frontend_reqs/training_data/{module_name}_pv_training_data.csv"):
        print("Loading existing dataset...")
        df = pd.read_csv(f"frontend_reqs/training_data/{module_name}_pv_training_data.csv")
        x = df[['irr', 'temp']].values
        y = df[['iph', 'isat', 'rsh', 'a']].values

    #else needs to generate dataset
    else:
        try:
            cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
            module = cec_modules['Prism_Solar_Technologies_Bi48_267BSTC']

            datasheet_conditions = (
                module['I_sc_ref'], 
                module['V_mp_ref'], 
                module['V_oc_ref'], 
                module['I_mp_ref'],
                module['N_s']
            )

            irradiances = []
            temps = []

            initial = 100
            for i in range(22):
                irradiances.append(initial + i*50)

            initial = 20
            for i in range(14):
                temps.append(initial + i*5)

            for irr in irradiances:
                for temp in temps:
                    test_cell = DataEntry(irr, temp, datasheet_conditions, module_name)

                    x0.append([irr, temp])
                    y0.append([test_cell.iph, test_cell.isat, test_cell.rsh, test_cell.a])

            x = np.array(x0)
            y = np.array(y0)

            # Combine inputs + outputs
            data = np.hstack((x, y))
            columns = ['irr', 'temp', 'iph', 'isat', 'rsh', 'a']

            #save to a csv file to avoid regenerating
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(f"frontend_reqs/training_data/{module_name}_pv_training_data.csv", index=False)
            print(f"Dataset saved to frontend_reqs/training_data/{module_name}_pv_training_data.csv")
        except Exception as e:
            print(f"Error - no datasheet present in pvlibs: {e}")

#returns the optimal model and scaler to adjust values
def create_optimal_ann():
    x, y = create_dataset()

    # Scale inputs and outputs
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2)

    structure = tuple([33] * 2)

    # Create ANN
    model = MLPRegressor(
        hidden_layer_sizes=structure,
        activation='tanh',
        solver='adam',
        max_iter=100000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_scaler, y_scaler