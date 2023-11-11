import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython import display

def select_action(state, actor_network, noise_process, action_space_low, action_space_high):
    actor_network.eval() # set to evaluation mode
    # state_tensor = torch.tensor(state, device=device)
    # print("state_tensor.shape: ", state_tensor.shape)
    # state_tensor_unsqueezed = state_tensor.unsqueeze(0) 
    deterministic_action = actor_network(state) #.cpu().detach().numpy()[0]
    # print("deterministic_action.shape: ", deterministic_action.shape)
    noisy_action = deterministic_action + noise_process.sample()
    # print("noisy_action.shape: ", noisy_action.shape)
    clipped_action = torch.clamp(noisy_action, action_space_low, action_space_high)
    # print("clipped_action.shape: ", clipped_action.shape)
    actor_network.train()  # set back to training mode
    return clipped_action


def soft_update(target_net, main_net, tau):
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * main_param.data)

def step_penalty_reward_transform(reward, step_penalty, current_step):
    return reward - step_penalty * current_step


def plot_reward(reward_list, episode_duration_list, is_ipython, run, average_window=100, show_result=False):
    """
    Plot the durations of episodes and sum of rewards as functions of time.
    """
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Rewards
    ax1 = plt.subplot(1, 2, 1)
    if not show_result:
        ax1.clear()
        ax1.set_title('Training... Rewards\n Number of runs: {}'.format(run))
    else:
        ax1.set_title('Result: Rewards')

    reward_list_t = torch.tensor(reward_list, dtype=torch.float)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Sum of reward per episode')
    line1, = ax1.plot(reward_list_t.numpy(), color='blue', label='Reward')
    
    # Plot the rolling average
    line2 = None
    if len(reward_list_t) >= average_window:
        means = reward_list_t.unfold(0, average_window, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(average_window-1), means))
        line2, = ax1.plot(means.numpy(), color='red', label='Average Reward')
    ax1.legend(loc='upper left')

    # Plot 2: Episode Durations
    ax2 = plt.subplot(1, 2, 2)
    if not show_result:
        ax2.clear()
        ax2.set_title('Training... Episode Durations\n Number of runs: {}'.format(run))
    else:
        ax2.set_title('Result: Episode Durations')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Duration of episode')
    line3, = ax2.plot(episode_duration_list, color='brown', label='Duration', linestyle='--')
    ax2.legend(loc='upper left')

    # Adjust spacing and display
    plt.tight_layout()
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



def plot_average_rewards(rewards, durations, average_window=20):
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.ylabel('Average Reward')
    plt.xlabel('Episode')
    plt.title('Average Reward vs Episode')
    if len(rewards) > average_window:
        rolling_mean = np.convolve(rewards, np.ones(average_window)/average_window, mode='valid')
        x_vals = range(average_window - 1, average_window - 1 + len(rolling_mean))
        plt.plot(x_vals, rolling_mean, color='red', linestyle='dashed')
        
    plt.subplot(1, 2, 2)
    plt.plot(durations)
    plt.ylabel('Average Duration')
    plt.xlabel('Episode')
    plt.title('Average Duration vs Episode')
    plt.tight_layout()
    plt.show()