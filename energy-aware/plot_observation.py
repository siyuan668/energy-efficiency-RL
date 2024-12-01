import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_observation():
    # Step 1: Load the CSV file into a DataFrame
    file_path = './logs_folder/observation.csv'  # Update with the actual path to your CSV file
    df = pd.read_csv(file_path)
    #tl_r, tl_g, tl_y, tl_current_phase, tl_phase_duration, tl_phase_remaining,bus_speed,bus_max_speed,bus_acceleration,bus_max_acceleration, bus_stop_duration, distance_to_bus_stop, int(is_in_bus_stop), remaining_time, distance_to_stopline, int(self.termination)
    column_names = [
        'tl_r', 'tl_g', 'tl_y', 'tl_current_phase', 'tl_phase_duration', 'tl_phase_remaining',
        'bus_speed', 'bus_max_speed', 'bus_acceleration', 'bus_max_acceleration',
        'bus_stop_duration', 'distance_to_bus_stop', 'int(is_in_bus_stop)', 'remaining_time',
        'distance_to_stopline', 'int(self.termination)'
    ]

    df.columns = column_names  # Assign the column names to the DataFrame

    save_folder = 'observation_plots'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # Step 2: Plot each column
    for column in df.columns:
        plt.figure(figsize=(10, 6))  # Set figure size
        plt.plot(df[column], label=column)
        plt.title(f'Plot of {column}')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.legend()
        plt.grid(True)

        # Step 3: Save the plot to an image file
        image_file_path = f'{column}_plot.png'
        plt.savefig(os.path.join(save_folder, image_file_path))
        plt.close()  # Close the plot to avoid overlap between different plots

    print("Plots saved successfully.")

if __name__ == "__main__":
    plot_observation()