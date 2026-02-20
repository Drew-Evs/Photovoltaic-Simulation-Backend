import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from predicting_parameters.single_cell import cell
import random
import pandas as pd
import os

import optuna

import time


def create_dataset():
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

    #sees if data exists and if not generates own 
    if os.path.exists("predicting_parameters/pv_training_data.csv"):
        print("Loading existing dataset...")
        df = pd.read_csv("predicting_parameters/pv_training_data.csv")
        x = df[['irr', 'temp']].values
        y = df[['iph', 'isat', 'rsh', 'a']].values
    else:
        print("Generating dataset...")
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
                test_cell = cell(irr, temp, datasheet_conditions, module_name)

                x0.append([irr, temp])
                y0.append([test_cell.iph, test_cell.isat, test_cell.rsh, test_cell.a])

        x = np.array(x0)
        y = np.array(y0)

        # Combine inputs + outputs
        data = np.hstack((x, y))
        columns = ['irr', 'temp', 'iph', 'isat', 'rsh', 'a']

        #save to a csv file to avoid regenerating
        df = pd.DataFrame(data, columns=columns)
        df.to_csv("pv_training_data.csv", index=False)
        print("Dataset saved to pv_training_data.csv")

    #adjust isat by log to make it better to predict
    y[:, 1] = np.log10(y[:, 1].astype(float))


    return x, y
    

#want to minimise the amount of time/accuracy based on number of layers
#attempting minimisation with optuna 
def objective(trial):
    layers = trial.suggest_int('layers', 1, 5)
    nodes = trial.suggest_int('nodes', 10, 100)

    structure = tuple([nodes] * layers)

    # Create ANN
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
    r2_loss = 1.0 - r2_score

    #measure time to calculate
    start = time.perf_counter()
    pred_scaled = model.predict(val_inputs_scaled)
    inference_time = time.perf_counter() - start

    pred = y_scaler.inverse_transform(pred_scaled)

    residual = (pred - val_calc_params) / val_calc_params
    rms_error = np.sqrt(np.mean(residual**2))

    time_penalty = inference_time * 10000

    # print(f"R residiual = {r2_loss}")
    # print(f"Rms residiual = {rms_error}")
    # print(f"Time residiual = {time_penalty}")

    #saving data totrial to graph
    trial.set_user_attr("r2_score", r2_score)
    trial.set_user_attr("time_penalty", time_penalty)

    return  r2_loss + time_penalty + rms_error

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



#used for making the dataset and finding optimal conditiions
if __name__ == "__main__":
    x,y = create_dataset()

    # # Scale inputs and outputs
    # X_scaler = StandardScaler()
    # y_scaler = StandardScaler()

    # X_scaled = X_scaler.fit_transform(x)
    # y_scaled = y_scaler.fit_transform(y)

    # # Split
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2)


    # import pvlib
    # cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
    # module = cec_modules['Prism_Solar_Technologies_Bi48_267BSTC']
    # module_name = 'Prism_Solar_Technologies_Bi48_267BSTC'

    # datasheet_conditions = (
    #     module['I_sc_ref'], 
    #     module['V_mp_ref'], 
    #     module['V_oc_ref'], 
    #     module['I_mp_ref'],
    #     module['N_s']
    # )

    # #generate randomly a dataset using the cell calculation
    # x0 = []
    # y0 = []

    # irr_bounds = (100, 1000)
    # temp_bounds = (25, 65)

    # num_validation_samples = 20
    # val_irr_list = []
    # val_temp_list = []
    # val_calc_params = []

    # print("Generating Samples")

    # #generate 20 validation items for optimisation
    # while len(val_calc_params) < num_validation_samples:
    #     try:
    #         irr = random.randint(100, 1000)
    #         temp = random.randint(*temp_bounds)
    #         test_cell = cell(irr, temp, datasheet_conditions, module_name)
            
    #         val_calc_params.append([test_cell.iph, test_cell.isat, test_cell.rsh, test_cell.a])
    #         val_irr_list.append(irr)
    #         val_temp_list.append(temp)
    #     except Exception as e:
    #         continue

    # print("Generated Samples")

    # val_calc_params = np.array(val_calc_params)
    # val_inputs = np.column_stack((val_irr_list, val_temp_list))
    # val_inputs_scaled = X_scaler.transform(val_inputs)

    # #finding the best performing values for layers and nodes
    # print("Using optuna optimisation")
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=300)

    # #get the best study
    # trial = study.best_trial
    # print("\nBest trial:")
    # print(f"  Value (Total Loss): {trial.value}")
    # print(f"  Best Params: {trial.params}")

    # completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # #extract the data from trials
    # extracted_data = []
    # for t in completed_trials:
    #     l = t.params['layers']
    #     n = t.params['nodes']
    #     r2 = t.user_attrs["r2_score"]
    #     actual_time_seconds = t.user_attrs["time_penalty"]/100
    #     extracted_data.append((l, n, actual_time_seconds, r2))

    # #sort by layers then nodes
    # extracted_data.sort(key=lambda x: (x[0], x[1]))
    # #seperate lists to plot
    # x_positions = [(l - 1) * 100 + n for l, n, time, r2 in extracted_data]
    # sorted_times = [x[2] for x in extracted_data]
    # sorted_r2 = [x[3] for x in extracted_data]

    # # We will use the index (0 to N) as the X-axis, and add custom labels
    # x_indices = range(len(extracted_data))

    # optimal_layers = trial.params['layers']
    # optimal_nodes = trial.params['nodes']

    # structure = tuple([optimal_nodes] * optimal_layers)

    # model = MLPRegressor(
    #     hidden_layer_sizes=structure,
    #     activation='tanh',
    #     solver='adam',
    #     max_iter=10000,
    #     random_state=42,
    #     early_stopping=True
    # )
    # model.fit(X_train, y_train)

    # #add the R² score (should be near to 0)
    # r2_score = model.score(X_test, y_test)

    # print(f'R2 score at best {r2_score}')

    # import matplotlib.pyplot as plt

    # # --- Plot the Graph ---
    # fig, ax1 = plt.subplots(figsize=(15, 7))

    # # Plot Actual Inference Time on the left Y-axis
    # color1 = 'tab:red'
    # ax1.set_ylabel('Inference Time (microseconds)', color=color1, fontsize=12, fontweight='bold')
    # ax1.scatter(x_positions, sorted_times, color=color1, alpha=0.6, s=25, label='Time (s)')

    # max_time = max(sorted_times)
    # ax1.set_ylim(0, max_time * 1.05)
    # ax1.tick_params(axis='y', labelcolor=color1)

    # # Create a second Y-axis for R² Score
    # ax2 = ax1.twinx()  
    # color2 = 'tab:blue'
    # ax2.set_ylabel('R² Score', color=color2, fontsize=12, fontweight='bold')  
    # ax2.scatter(x_positions, sorted_r2, color=color2, marker='x', alpha=0.7, s=25, label='R²')
    # ax2.tick_params(axis='y', labelcolor=color2)

    # # --- Format the X-Axis to strictly show Even Blocks and Nodes ---
    # major_ticks = []
    # major_labels = []

    # # Loop through our 5 possible layers to draw the grid
    # for layer in range(1, 6):
    #     base_x = (layer - 1) * 100
        
    #     # Place the main layer label perfectly in the center of the 100-width block
    #     major_ticks.append(base_x + 50)
    #     major_labels.append(f"{layer} Layer(s)")
        
    #     # Draw a strict vertical divider between layer blocks
    #     if layer > 1:
    #         ax1.axvline(x=base_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        
    #     # Add text annotations at the bottom to show the node bounds (10 to 100) inside each block
    #     ax1.text(base_x + 10, -0.06, '10 nodes', transform=ax1.get_xaxis_transform(), color='dimgray', fontsize=9, ha='center')
    #     ax1.text(base_x + 100, -0.06, '100 nodes', transform=ax1.get_xaxis_transform(), color='dimgray', fontsize=9, ha='center')

    # # Apply the major ticks (Layer names)
    # ax1.set_xticks(major_ticks)
    # ax1.set_xticklabels(major_labels, fontweight='bold', fontsize=12)

    # # Set the padding (spacing) using tick_params instead
    # ax1.tick_params(axis='x', pad=25)

    # # Pad the X-axis so the dots don't clip on the edges
    # ax1.set_xlim(-10, 520) 
    # ax1.set_xlabel('Network Architecture (Subdivided by Node Count)', fontsize=11, color='dimgray', labelpad=30)

    # # Add a grid for the Y axes to make reading values easier
    # ax1.grid(axis='y', color='tab:red', linestyle=':', alpha=0.2)
    # ax2.grid(axis='y', color='tab:blue', linestyle=':', alpha=0.2)

    # plt.title('Performance vs. Architecture: Real Inference Time and R² Score', fontsize=15, pad=15)
    # fig.tight_layout()

    # # Save the plot
    # plt.savefig('optuna_architecture_graph.png', dpi=300)
    # print("Graph saved as 'optuna_architecture_graph.png'")