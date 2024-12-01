import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_reward():
    # Step 1: Load the CSV file into a DataFrame
    file_path = './logs_folder/reward.csv'  # Update with the actual path to your CSV file
    df = pd.read_csv(file_path)

    column_names = ['tl_reward', 'speed_reward', 'energy_reward', 'total_reward', 'self.total_energy_comsumption', 'self.total_running_distance_per_sec', 'self.total_running_distance']
    df.columns = column_names  # Assign the column names to the DataFrame

    save_folder = 'reward_plots'
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
        print(f'plot saved to {os.path.join(save_folder, image_file_path)}')
        plt.close()  # Close the plot to avoid overlap between different plots

    print("Plots saved successfully.")

if __name__ == "__main__":
    plot_reward()