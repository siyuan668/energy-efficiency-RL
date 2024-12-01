import json
import matplotlib.pyplot as plt
import os

def plot_loss(file_name):
    # Read the file
    #folder_name = "./logs_folder/"
    #file_name = 'train_log_lr_0.05_2024_11_15_21_14_41'
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Extract the required data
    total_loss = []
    vf_loss_unclipped = []
    policy_loss = []
    curr_kl_coeff = []
    mean_kl_loss = []
    entropy = []
    iterations = []

    for i, entry in enumerate(data, 1):
        learner_data = entry['learners']['default_policy']
        total_loss.append(learner_data['total_loss'])
        vf_loss_unclipped.append(learner_data['vf_loss_unclipped'])
        policy_loss.append(learner_data['policy_loss'])
        curr_kl_coeff.append(learner_data['curr_kl_coeff'])
        mean_kl_loss.append(learner_data['mean_kl_loss'])
        entropy.append(learner_data['entropy'])
        iterations.append(i)

    # Create a figure with 6 subplots in a 2x3 layout
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))

    # Plot Total Loss
    ax1.plot(iterations, total_loss, 'b-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss over Iterations')
    ax1.grid(True)

    # Plot VF Loss Unclipped
    ax2.plot(iterations, vf_loss_unclipped, 'r-')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('VF Loss Unclipped')
    ax2.set_title('VF Loss Unclipped over Iterations')
    ax2.grid(True)

    # Plot Policy Loss
    ax3.plot(iterations, policy_loss, 'g-')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Policy Loss')
    ax3.set_title('Policy Loss over Iterations')
    ax3.grid(True)

    # Plot Current KL Coefficient
    ax4.plot(iterations, curr_kl_coeff, 'm-')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Current KL Coefficient')
    ax4.set_title('Current KL Coefficient over Iterations')
    ax4.grid(True)

    # Plot Mean KL Loss
    ax5.plot(iterations, mean_kl_loss, 'c-')
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Mean KL Loss')
    ax5.set_title('Mean KL Loss over Iterations')
    ax5.grid(True)

    # Plot Entropy
    ax6.plot(iterations, entropy, 'y-')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Entropy')
    ax6.set_title('Entropy over Iterations')
    ax6.grid(True)

    # Adjust layout and show plot
    plt.tight_layout()

    # Save the plot as a PNG image
    plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    file_name = './logs_folder/train_log_lr_0.05_2024_11_15_21_14_41.csv'
    plot_loss(file_name)
