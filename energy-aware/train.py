from ray.rllib.algorithms.ppo import PPOConfig

from ray.tune.registry import register_env
from environment import MyEnv
from ray.rllib.utils.typing import ModelConfigDict
from pprint import pprint
import os
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
import torch.nn as nn

import pathlib
import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule

from typing import Any, Dict
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec, RLModuleConfig
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule

from datetime import datetime
import json
from ray.rllib.utils.filter import MeanStdFilter

from plot_loss import plot_loss
from plot_observation import plot_observation
from plot_reward import plot_reward
from plot_distance_time import plot_distance
import json
import shutil



def compute_action():
    env = gym.make("CartPole-v1")

    # Create only the neural network (RLModule) from our checkpoint.
    rl_module = RLModule.from_checkpoint(
        pathlib.Path(best_checkpoint.path) / "learner_group" / "learner" / "rl_module"
    )["default_policy"]

    episode_return = 0
    terminated = truncated = False

    obs, info = env.reset()

    while not terminated and not truncated:
        # Compute the next action from a batch (B=1) of observations.
        torch_obs_batch = torch.from_numpy(np.array([obs]))
        action_logits = rl_module.forward_inference({"obs": torch_obs_batch})[
            "action_dist_inputs"
        ]
        # we'll have to sample an action or use the max-likelihood one).
        action = torch.argmax(action_logits[0]).numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

    print(f"Reached episode return of {episode_return}.")

class MyTorchModule(TorchRLModule):
    def __init__(self, config: RLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):
        print(self.config)
        input_dim = 4 #self.config.obs_num#4
        hidden_dim =64 #self.config.model_config_dict["fcnet_hiddens"][0] #64
        output_dim = 2 #self.config.action_num#2

        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.input_dim = input_dim

    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        action_logits = self.policy(batch["obs"])
        return {"action_dist": torch.distributions.Categorical(logits=action_logits)}

def train(total_step = 5):

    log_names = ['./logs_folder/reward.csv', './logs_folder/action.csv', './logs_folder/observation.csv', 
                 './logs_folder/dist_sec_step.csv', './logs_folder/dist_sec_step.csv']

    for log_name in log_names:
        if os.path.exists(log_name):
            os.remove(log_name)
    #return

    env= MyEnv()
    register_env("my_env",lambda _:env)

    config = PPOConfig()
    config.api_stack(enable_rl_module_and_learner=True,
                     enable_env_runner_and_connector_v2=True,
                    )
    config.environment("my_env")
    config.env_runners(num_env_runners=1)
    config.framework("torch")
    
    #load config
    # Load the JSON file
    with open('./rl_training/config.json', 'r') as file:
        system_config = json.load(file)

    config.seed = 42 #added randomness

    lr =system_config['train']['lr'] 
    gamma = system_config['train']['gamma']
    kl_coeff = system_config['train']['kl_coeff']
    train_batch_size_per_learner = system_config['train']['train_batch_size_per_learner']
    fcnet_hiddens = system_config['train']['fcnet_hiddens']
    fcnet_activation = system_config['train']['fcnet_activation']
    use_lstm = system_config['train']['use_lstm']
    use_attention = system_config['train']['use_attention']
    config.training(
        gamma=gamma, #0.9
        lr=lr, #0.0003
        kl_coeff=kl_coeff, #0.3
        train_batch_size_per_learner=train_batch_size_per_learner,  #128
        model={
        "fcnet_hiddens": fcnet_hiddens, # [64, 128,128, 64],
        "fcnet_activation": fcnet_activation, #"tanh",#"relu",
        "use_lstm": use_lstm, #True,
        "use_attention": use_attention, #True
        }
        
        )

    #new added
    config.rl_module(rl_module_spec=generate_RL_Module())

    algo = config.build()

    #steps = 0
    cur_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    training_result_folder = './logs_folder/training_result_folder_'+ cur_time_str
    if not os.path.exists(training_result_folder):
        os.makedirs(training_result_folder)
    checkpoint_dir_base = 'mycheckpoint_lr_'+str(lr)
    train_log_filename = training_result_folder+'/train_log_lr_'+str(lr) +'_' + cur_time_str + '.txt'
    print(f'train_log_filename: {train_log_filename}')
    #while not terminated and not truncated:
    results = []
    #total_step = 5
    for steps in range(total_step):
        #steps += 1
        result = algo.train()
        result.pop("config")
        pprint(result)

        if not os.path.exists(train_log_filename):
            with open(train_log_filename, 'w') as file:
                json.dump([], file, indent=4)  # Initialize the file with an empty list

        # Open the file in 'r+' mode to read and write
        with open(train_log_filename, 'r+') as file:
            try:
                data = json.load(file)  # Read the existing content
            except json.JSONDecodeError:  # Handle empty or invalid JSON
                data = []

            # Append the new result to the list
            data.append(result)

            # Move back to the beginning of the file and write the updated data
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()  # Ensure no extra data remains after the write

        if (steps+1) % 20 == 0: #TODO: don't save while fine tuning hyperparameters
            checkpoint_dir = f'{checkpoint_dir_base}_{steps+1}'
            checkpoint_dir = os.path.join(training_result_folder, checkpoint_dir)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                print(f"Directory '{checkpoint_dir}' created successfully!")
            checkpoint_dir = algo.save_to_path(checkpoint_dir)
            print(f"Checkpoint saved in directory {checkpoint_dir}")
        
        if steps==total_step-1 and (steps+1)%20!=0:#save the last step
            checkpoint_dir = f'{checkpoint_dir_base}_{steps+1}'
            checkpoint_dir = os.path.join(training_result_folder, checkpoint_dir)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                print(f"Directory '{checkpoint_dir}' created successfully!")
            checkpoint_dir = algo.save_to_path(checkpoint_dir)
            print(f"Checkpoint saved in directory {checkpoint_dir}")


    #visualization
    plot_observation()
    plot_reward()
    plot_distance()
    plot_loss(train_log_filename)

    #move logs, plots to training_result_folder
    src_file_list = ['./logs_folder/dist_sec_step.csv', './logs_folder/observation.csv', './logs_folder/reward.csv', 
                    './logs_folder/action.csv','./observation_plots','./reward_plots']
    
    dst_file_list = [training_result_folder+'/dist_sec_step.csv', training_result_folder+'/observation.csv', training_result_folder+'/reward.csv', 
                    training_result_folder+'/action.csv',training_result_folder+'/observation_plots',training_result_folder+'/reward_plots']
    
    for src, dst in zip(src_file_list, dst_file_list):
        try:
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}")
        except Exception as e:
            print(f"Error moving {src} to {dst}: {e}")

    shutil.copy('./rl_training/config.json', training_result_folder+'/config.json')


#new added
def generate_RL_Module():
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
    return RLModuleSpec(module_class=PPOTorchRLModule)
#new added
def evaluate_policy(path_to_checkpoint):
    env = MyEnv()
    register_env("my_env",lambda _:env)

    from ray.rllib.algorithms.algorithm import Algorithm

    loaded_algo = Algorithm.from_checkpoint(path_to_checkpoint)
    loaded_algo.train()

if __name__ == "__main__":

    path_to_checkpoint = './logs_folder/training_result_folder_2024_11_26_11_24_50/mycheckpoint_lr_0.0003_20'
    evaluate_policy(path_to_checkpoint)