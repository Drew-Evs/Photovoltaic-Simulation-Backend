import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_power_data(input_csv):
    try:
        print(f"Loading data from {input_csv}...")
        df = pd.read_csv(input_csv)

        # Ensure the columns exist
        if 'time' not in df.columns or 'power' not in df.columns:
            print("Error: Could not find 'time' and 'power' columns. Available columns are:", df.columns)
            return
        
        # Keep only data from 80,000 seconds onwards
        df = df[df['time'] >= 160000].copy()
        
        # Convert the time to hours, setting 80,000 seconds to Hour 0
        df['time_hours'] = (df['time'] - 160000) / 3600

        # Safety check in case the CSV doesn't go up to 80,000 yet
        if df.empty:
            print("Error: No data found at or after 80,000 seconds.")
            return

        # Create the plot
        plt.figure(figsize=(12, 6)) 
        
        # Plot time vs power. (Swapped 'time' to 'time_hours')
        plt.plot(df['time_hours'], df['power'], marker='o', linestyle='-', color='b', markersize=2, label='MPPT Power Output', zorder=1)

        # --- Check for shaded_substrings and plot the changes ---
        if 'shaded_substrings' in df.columns:
            df['state_changed'] = df['shaded_substrings'] != df['shaded_substrings'].shift(1)
            # Filter to rows where a change occurred (ignoring the very first row)
            changes = df[(df['state_changed']) & (df.index > df.index[0])]

            # Dictionary to assign distinct shapes and colors to each state
            state_styles = {
                0: {'marker': 'o', 'color': 'green',      'label': '0 Shaded'},  # Green Circle
                1: {'marker': '^', 'color': 'goldenrod',  'label': '1 Shaded'},  # Yellow Triangle
                2: {'marker': 'D', 'color': 'darkorange', 'label': '2 Shaded'},  # Orange Diamond
                3: {'marker': 's', 'color': 'red',        'label': '3 Shaded'}   # Red Square
            }

            added_labels = set()
            
            # Calculate the range of the Y-axis to dynamically offset the text so it doesn't overlap
            y_range = df['power'].max() - df['power'].min()
            if y_range == 0: y_range = 10 

            # Loop through each change and draw markers
            for idx, row in changes.iterrows():
                change_time = row['time_hours']
                change_power = row['power']
                substrings = int(row['shaded_substrings'])

                # Get the specific color and shape for this state
                style = state_styles.get(substrings, {'marker': 'X', 'color': 'black', 'label': 'Unknown'})

                # Draw a vertical dashed line matching the color of the state
                plt.axvline(x=change_time, color=style['color'], linestyle='--', alpha=0.3, zorder=0)
                
                # Highlight the exact data point with the unique marker
                label = style['label'] if style['label'] not in added_labels else ""
                plt.plot(change_time, change_power, marker=style['marker'], color=style['color'], 
                         markersize=8, linestyle='None', label=label, zorder=3)
                added_labels.add(style['label'])
                
                # --- NEW: Add text directly above the marker ---
                plt.text(change_time, change_power + (y_range * 0.04), 
                         f"{substrings} Shaded", 
                         color=style['color'], fontsize=10, fontweight='bold', 
                         ha='center', va='bottom', zorder=4)
        else:
            print("Notice: 'shaded_substrings' column not found. Skipping change markers.")

        # Add some padding to the top of the graph so the floating text doesn't get clipped
        plt.margins(y=0.15)

        # Add labels, title, and grid
        plt.title('Solar Panel Power Output Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Elapsed Time (Hours)', fontsize=12) # Updated label to Hours
        plt.ylabel('Power (Watts)', fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Display the graph!
        print("Graph generated successfully.")
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{csv_filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Load the CSV data into a Pandas DataFrame
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(BASE_DIR, "simulation_data")

    #ensure folder exists
    os.makedirs(target_dir, exist_ok=True)

    #finds input and output file
    input_csv = os.path.join(target_dir, "Saved_Power_Time_Data.csv")

    plot_power_data(input_csv)