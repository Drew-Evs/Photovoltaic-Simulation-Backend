from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import os
import pvlib
import numpy as np
from pathlib import Path

from scipy.optimize import fsolve, least_squares

from predicting_parameters import refactored_prediction

#used to build the database if needed
class DataEntry():
    def __init__(self, irr, temp, datasheet_conditions, module_name, specs):
        #datasheet conditions and parameter modelling
        self.isc, self.vmp, self.voc, self.imp, self.Ns = datasheet_conditions
        self.voc_per_cell = self.voc/self.Ns
        actual_params = refactored_prediction.getting_parameters_specs(temp, irr, specs)
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

    def get_params(self):
        return self.a, self.iph, self.isat, self.rs, self.rsh

def create_dataset(module_name, specs):
    #generate randomly a dataset using the cell calculation
    x0 = []
    y0 = []

    #save at absolute root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent 
    target_dir = PROJECT_ROOT / "training_data"
    os.makedirs(target_dir, exist_ok=True)
    filepath = target_dir / f"{module_name}_pv_training_data.csv"
    
    #sees if data exists and if not generates own 
    if os.path.exists(filepath):
        print("Loading existing dataset...")
        df = pd.read_csv(filepath)
        x = df[['irr', 'temp']].values
        y = df[['iph', 'isat', 'rsh', 'a']].values

    #else needs to generate dataset
    else:        
        isc = specs['I_sc']
        vmp = specs['V_mp']
        voc = specs['V_oc']
        imp = specs['I_mp']
        Ns = specs['N_s']

        datasheet_conditions = isc, vmp, voc, imp, Ns

        irradiances = []
        temps = []

        initial = 100
        for i in range(22):
            irradiances.append(initial + i*50)

        initial = 10
        for i in range(7):
            temps.append(initial + i*10)

        for irr in irradiances:
            for temp in temps:
                test_cell = DataEntry(irr, temp, datasheet_conditions, module_name, specs)

                x0.append([irr, temp])
                y0.append([test_cell.iph, test_cell.isat, test_cell.rsh, test_cell.a])

        x = np.array(x0)
        y = np.array(y0)

        # Combine inputs + outputs
        data = np.hstack((x, y))
        columns = ['irr', 'temp', 'iph', 'isat', 'rsh', 'a']

        os.makedirs(filepath, exist_ok=True)

        #save to a csv file to avoid regenerating
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")

    #adjust isat by log to make it better to predict
    y[:, 1] = np.log10(y[:, 1].astype(float))

    return x, y


#returns the optimal model and scaler to adjust values
def create_optimal_ann(module_name, specs):
    x, y = create_dataset(module_name, specs)

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

#used for making the dataset and finding optimal conditiions
if __name__ == "__main__":
    x,y = create_dataset()

    # Scale inputs and outputs
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2)


    import pvlib
    cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
    module = cec_modules['Prism_Solar_Technologies_Bi48_267BSTC']
    module_name = 'Prism_Solar_Technologies_Bi48_267BSTC'

    datasheet_conditions = (
        module['I_sc_ref'], 
        module['V_mp_ref'], 
        module['V_oc_ref'], 
        module['I_mp_ref'],
        module['N_s']
    )

    #generate randomly a dataset using the cell calculation
    x0 = []
    y0 = []

    irr_bounds = (100, 1000)
    temp_bounds = (25, 65)

    num_validation_samples = 20
    val_irr_list = []
    val_temp_list = []
    val_calc_params = []

    print("Generating Samples")

    #generate 20 validation items for optimisation
    while len(val_calc_params) < num_validation_samples:
        try:
            irr = random.randint(100, 1000)
            temp = random.randint(*temp_bounds)
            test_cell = cell(irr, temp, datasheet_conditions, module_name)
            
            val_calc_params.append([test_cell.iph, test_cell.isat, test_cell.rsh, test_cell.a])
            val_irr_list.append(irr)
            val_temp_list.append(temp)
        except Exception as e:
            continue

    print("Generated Samples")

    val_calc_params = np.array(val_calc_params)
    val_inputs = np.column_stack((val_irr_list, val_temp_list))
    val_inputs_scaled = X_scaler.transform(val_inputs)

    #finding the best performing values for layers and nodes
    print("Using optuna optimisation")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=300)

    #get the best study
    trial = study.best_trial
    print("\nBest trial:")
    print(f"  Value (Total Loss): {trial.value}")
    print(f"  Best Params: {trial.params}")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    #extract the data from trials
    extracted_data = []
    for t in completed_trials:
        l = t.params['layers']
        n = t.params['nodes']
        r2 = t.user_attrs["r2_score"]
        actual_time_seconds = t.user_attrs["time_penalty"]/100
        extracted_data.append((l, n, actual_time_seconds, r2))

    #sort by layers then nodes
    extracted_data.sort(key=lambda x: (x[0], x[1]))
    #seperate lists to plot
    x_positions = [(l - 1) * 100 + n for l, n, time, r2 in extracted_data]
    sorted_times = [x[2] for x in extracted_data]
    sorted_rms = [x[3] for x in extracted_data]

    # We will use the index (0 to N) as the X-axis, and add custom labels
    x_indices = range(len(extracted_data))

    optimal_layers = trial.params['layers']
    optimal_nodes = trial.params['nodes']

    structure = tuple([optimal_nodes] * optimal_layers)

    model = MLPRegressor(
        hidden_layer_sizes=structure,
        activation='tanh',
        solver='adam',
        max_iter=10000,
        random_state=42,
        early_stopping=True
    )
    model.fit(X_train, y_train)

    #add the R² score (should be near to 0)
    r2_score = model.score(X_test, y_test)

    print(f'R2 score at best {r2_score}')

    import matplotlib.pyplot as plt

    # --- Plot the Graph ---
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot Actual Inference Time on the left Y-axis
    color1 = 'tab:red'
    ax1.set_ylabel('Inference Time (microseconds)', color=color1, fontsize=12, fontweight='bold')
    ax1.scatter(x_positions, sorted_times, color=color1, alpha=0.6, s=25, label='Time (s)')

    max_time = max(sorted_times)
    ax1.set_ylim(0, max_time * 1.05)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Create a second Y-axis for R² Score
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    
    # Update label and data to use sorted_rms
    ax2.set_ylabel('RMS Error (Fractional)', color=color2, fontsize=12, fontweight='bold')  
    ax2.scatter(x_positions, sorted_rms, color=color2, marker='x', alpha=0.7, s=25, label='RMS')
    ax2.tick_params(axis='y', labelcolor=color2)


    # --- Format the X-Axis to strictly show Even Blocks and Nodes ---
    major_ticks = []
    major_labels = []

    # Loop through our 5 possible layers to draw the grid
    for layer in range(1, 6):
        base_x = (layer - 1) * 100
        
        # Place the main layer label perfectly in the center of the 100-width block
        major_ticks.append(base_x + 50)
        major_labels.append(f"{layer} Layer(s)")
        
        # Draw a strict vertical divider between layer blocks
        if layer > 1:
            ax1.axvline(x=base_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Add text annotations at the bottom to show the node bounds (10 to 100) inside each block
        ax1.text(base_x + 10, -0.06, '10 nodes', transform=ax1.get_xaxis_transform(), color='dimgray', fontsize=9, ha='center')
        ax1.text(base_x + 100, -0.06, '100 nodes', transform=ax1.get_xaxis_transform(), color='dimgray', fontsize=9, ha='center')

    # Apply the major ticks (Layer names)
    ax1.set_xticks(major_ticks)
    ax1.set_xticklabels(major_labels, fontweight='bold', fontsize=12)

    # Set the padding (spacing) using tick_params instead
    ax1.tick_params(axis='x', pad=25)

    # Pad the X-axis so the dots don't clip on the edges
    ax1.set_xlim(-10, 520) 
    ax1.set_xlabel('Network Architecture (Subdivided by Node Count)', fontsize=11, color='dimgray', labelpad=30)

    # Add a grid for the Y axes to make reading values easier
    ax1.grid(axis='y', color='tab:red', linestyle=':', alpha=0.2)
    ax2.grid(axis='y', color='tab:blue', linestyle=':', alpha=0.2)

    # Update Title
    plt.title('Performance vs. Architecture: Real Inference Time and RMS Error', fontsize=15, pad=15)
    fig.tight_layout()

    # Save the plot
    plt.show()