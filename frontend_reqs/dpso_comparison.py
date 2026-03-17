import numpy as np
import matplotlib.pyplot as plt
import pvlib
import time
import random

# Assuming your imports look something like this in your main file:
from current_module import Module as C_Module
from refactored_whole_module import Module as V_Module
from current_dpso import C_DPSO_MPPT
from DPSO_MPPT import DPSO_MPPT

def power_over_time(v_module, c_module):
    # Initialize both trackers
    v_tracker = DPSO_MPPT(v_module.d - 1, v_module, 0, v_module.voc)
    c_tracker = C_DPSO_MPPT(c_module.d - 1, c_module, c_module.isc)

    # State of simulation at start
    total_steps = 15
    current_shade_level = 500.0
    target_shade_level = 500.0
    shaded_cells = set()

    print("Starting continuous dual-MPPT simulation...")

    # Trackers for plotting
    v_tracker_powers = []
    c_tracker_powers = []

    # Trackers for execution time
    v_total_time = 0.0
    c_total_time = 0.0

    for step in range(total_steps):
        # Decide if big step or little step 
        event_roll = random.random()

        if event_roll < 0.1:
            # Assume a large hard shadow - clouds
            if random.random() < 0.5:
                current_shade_level = random.uniform(100, 400) 
                target_shade_level = current_shade_level
                
                # Randomly pick a large block of cells
                block_size = random.randint(5, 20)
                shaded_start = random.randint(0, v_module.Ns - block_size)
                
                # Overwrite the shaded_cells set with this new block
                shaded_cells = set(range(shaded_start, shaded_start + block_size))
                
            # Or suddenly sunny
            else:
                current_shade_level = 1000.0 
                target_shade_level = 1000.0
                
                # Clear all shaded cells so the whole panel is in full sun
                shaded_cells.clear()

        # If not do gradual shade 
        else:
            # Drift towards different shade target
            if abs(target_shade_level - current_shade_level) < 10:
                target_shade_level = random.uniform(200, 1000)

            current_shade_level += (target_shade_level - current_shade_level) * 0.15

        # Randomly 0 to 6 cells get toggled
        num_to_toggle = random.randint(0, 6)
        cells_to_toggle = random.sample(range(v_module.Ns), num_to_toggle)

        # Toggle if unshaded to shaded and vice versa
        for cell in cells_to_toggle:
            if cell in shaded_cells:
                shaded_cells.remove(cell)
            else:
                shaded_cells.add(cell)

        # Generate the irradiance array for this step
        current_irr = np.full(v_module.Ns, 1000.0)
        for cell in shaded_cells:
            current_irr[cell] = current_shade_level

        # Apply conditions to both trackers/modules
        v_tracker.set_module_conditions(irr_array=current_irr)
        c_tracker.set_module_conditions(irr_array=current_irr)

        # --- RUN AND TIME V-TRACKER ---
        start_v_time = time.perf_counter()
        _, v_pmp = v_tracker.track_mpp()
        v_total_time += (time.perf_counter() - start_v_time)
        
        # --- RUN AND TIME C-TRACKER ---
        start_c_time = time.perf_counter()
        _, c_pmp = c_tracker.track_mpp()
        c_total_time += (time.perf_counter() - start_c_time)
        
        v_tracker_powers.append(v_pmp)
        c_tracker_powers.append(c_pmp)

        # Force state to local for the next loop (as per original logic)
        v_tracker.state = 'Local'
        c_tracker.state = 'Local'

        print(f"Step {step:<3}: Irr {current_shade_level:<4.0f} | V-Power: {v_pmp:<6.2f}W | C-Power: {c_pmp:<6.2f}W")

    # --- PRINT TIMING RESULTS ---
    print("\n" + "="*40)
    print("SIMULATION COMPLETE - TIMING RESULTS")
    print("="*40)
    print(f"Voltage-Based DPSO Total Time: {v_total_time:.4f} seconds")
    print(f"Current-Based DPSO Total Time: {c_total_time:.4f} seconds")
    
    if c_total_time > 0 and v_total_time > 0:
        if v_total_time < c_total_time:
            print(f"V-DPSO was {(c_total_time/v_total_time):.2f}x faster.")
        else:
            print(f"C-DPSO was {(v_total_time/c_total_time):.2f}x faster.")
    print("="*40 + "\n")

    # --- Plotting the V-Tracker vs C-Tracker Power ---
    plt.figure(figsize=(12, 6))
    
    # Plot V-DPSO
    plt.plot(range(total_steps), v_tracker_powers, label='Voltage-Based DPSO', color='dodgerblue', linewidth=2, zorder=2)
    
    # Plot C-DPSO (Dashed so you can see if they overlap perfectly)
    plt.plot(range(total_steps), c_tracker_powers, label='Current-Based DPSO', color='darkorange', linestyle='--', linewidth=2, zorder=3)

    # Fill the area between them to highlight discrepancies
    plt.fill_between(range(total_steps), v_tracker_powers, c_tracker_powers, color='red', alpha=0.3, label='Tracking Divergence')

    plt.title('V-DPSO vs C-DPSO Hybrid MPPT Under Dynamic Shading')
    plt.xlabel('Simulation Step')
    plt.ylabel('Power (Watts)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    plot_filename = 'tracker_head_to_head_comparison.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Performance graph successfully saved as '{plot_filename}'")
    plt.close()
    
if __name__ == "__main__":
    cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
    module = cec_modules['Prism_Solar_Technologies_Bi48_267BSTC']
    datasheet_conditions = (
        module['I_sc_ref'], 
        module['V_mp_ref'], 
        module['V_oc_ref'], 
        module['I_mp_ref'],
        module['N_s']
    )

    v_module = V_Module(datasheet_conditions, 'Prism_Solar_Technologies_Bi48_267BSTC')
    c_module = C_Module(datasheet_conditions, 'Prism_Solar_Technologies_Bi48_267BSTC')

    # Run the test
    power_over_time(v_module, c_module)