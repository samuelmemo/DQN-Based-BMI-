import numpy as np
import matplotlib.pyplot as plt
import torch
from models import DQNAgent
import os 
import scipy.io 

def Reaching_target_xy(n_targets, radius,center):
    angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)
    targets = np.array([[center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)] for angle in angles])
    return targets

def action_selection(max_out, n_possible_actions, epsilon):

    """Epsilon-greedy action selection.
    Args:
        max_out (np.array): Q-values for each action.
        n_possible_actions (int): Number of possible actions.
        epsilon (float): Probability of choosing a random action.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(n_possible_actions)
    else:
        return np.argmax(max_out)
    
def cursor_Next_XY(current_cursor_pos,selected_action, n_targets,
                   center, radius):

    """ Given the current cursor position, find the next target position in a 
    circular arrangement
     Args:

     current_cursor_pos (np.array): Current cursor position [x, y].
        selected_action (int): Index of the selected action/target.
        n_targets (int): Number of targets arranged in a circle.
        center (tuple): Center coordinates of the circle (x_center, y_center).
        radius (float): Radius of the circle on which targets are placed.
     
     """

     # Angles for all possible actions (exclude duplicate 2π)
    angles = np.linspace(0, 2*np.pi - 2*np.pi/n_targets, n_targets)
    # Possible target deltas per action (mirrors MATLAB: (cos+cx, sin+cy)*radius)
    possible_xy = np.round(
        np.column_stack([
            center[0] + np.cos(angles)* radius,
            center[1] + np.sin(angles)* radius
        ]), decimals=3)
    #
    cursor_next_xy = np.asarray(current_cursor_pos) + possible_xy[selected_action]
    return cursor_next_xy



def channel_standardizer(X_train, eps=1e-8):
    # mean/std over trials and time, per channel
    mu = X_train.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
    sd = X_train.std(axis=(0, 2), keepdims=True) 
    X =    (X_train - mu) / (sd + eps)
    return X



def train_agent(normalized_data, labels, epochs, montecarlo_runs, chans, batch_size, samples, kernLength, model_name, 
                n_targets, reaching_target_xy, reaching_center, 
                reaching_radius, reaching_distance_threshold, reward_success, reward_failure):

    n_trials = normalized_data.shape[0]
    
    # Use NaN instead of zeros to distinguish unrun epochs
    success_rate = np.full((montecarlo_runs, epochs), np.nan)
    epochs_completed = np.zeros(montecarlo_runs, dtype=int)

    for r in range(montecarlo_runs):
        print(f"Starting Monte Carlo run {r+1}/{montecarlo_runs}")
        agent = DQNAgent(nb_classes=n_targets, chans=chans, samples=samples, kernLength=kernLength, 
                         model_name=model_name, batch_size= batch_size)
    
        # Shuffle trials for each run
        trial_indices = np.random.permutation(n_trials)
        
        for epoch in range(epochs):
            success_count = 0
            
            for i in range(n_trials):
                trial_idx = trial_indices[i]
                state = normalized_data[trial_idx]
                target_label = labels[trial_idx] - 1 # 0-based index
                target_pos = reaching_target_xy[target_label]
                
                # Start cursor at center
                cursor_pos = reaching_center

                # Select action
                action = agent.select_action(state)
                
                # Execute action (move cursor)
                next_cursor_pos = cursor_Next_XY(cursor_pos, action, n_targets, reaching_center, reaching_radius)
                
                # Calculate reward
                distance = np.sqrt((next_cursor_pos[0] - target_pos[0])**2 + (next_cursor_pos[1] - target_pos[1])**2)
                if distance < reaching_distance_threshold:
                    reward = reward_success
                    success_count += 1
                else:
                    reward = reward_failure
                
                # Store transition
                done = True
                agent.memory.push(state, action, reward, state, done)
                
                # Train agent
                agent.update()
            
            agent.update_target_network()
            
            
            success_rate[r, epoch] = success_count / n_trials
            epochs_completed[r] = epoch + 1  # Track completed epochs
            print(f"  Epoch {epoch+1}/{epochs} - Success Rate: {success_rate[r, epoch]*100:.2f}%")

            # Stopping criterion:
            # Learning stops when the average success rates over the last 
            # three epochs, including the current epoch, does not
            # increase more than a set threshold of 0.01.
            if epoch >= 3:
                recent_avg = np.mean(success_rate[r, epoch-2:epoch+1])
                previous_avg = np.mean(success_rate[r, epoch-3:epoch])
                if recent_avg - previous_avg < 0.01:
                    print(f"Stopping early at epoch {epoch+1} due to convergence.")
                    break

    
    

    print("Training completed.")

    return agent, success_rate


def plot_success_rate(success_rate,dataset,model,
                      subject,session,
                      low_band, high_band, save=False):
    
    plt.figure(figsize=(10, 6))
    # Use nanmean/nanstd to ignore NaN values (unrun epochs)
    mean_success_rate = np.nanmean(success_rate, axis=0)
    std_success_rate = np.nanstd(success_rate, axis=0)
    
    # Only plot epochs that have at least one run with data
    valid_epochs = ~np.all(np.isnan(success_rate), axis=0)
    actual_epochs = np.sum(valid_epochs)
    mean_success_rate = mean_success_rate[valid_epochs]
    std_success_rate = std_success_rate[valid_epochs]

    plt.plot(range(1, actual_epochs + 1), mean_success_rate, label='Mean Success Rate')
    plt.fill_between(range(1, actual_epochs + 1), mean_success_rate - std_success_rate, mean_success_rate + std_success_rate, alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.title('EEGNet DQN Success Rate over Epochs')
    plt.legend(fontsize='small', loc='best')
    plt.grid(True)
    if save:
        folder_fig =f'{model}_{dataset}_success_rate_plots_band_{low_band}_{high_band}'
        if not os.path.exists(folder_fig):
            os.makedirs(folder_fig)
        plt.savefig(f'{folder_fig}/success_rate__{subject}_{session}.png')
    plt.show()

def load_jenna_data(high_cut=30, directory='OneDrive_1_12-19-2025'):

    n_channels =21
    
    
    jenna_dataset = {}
   
    for subdir in os.listdir(directory):
        if subdir.startswith('FREEFORMSubject'):
            i_path = os.path.join(directory, subdir, 'Data_from-0.5_to0', 'Data_EEGfeatureRAW.mat')
        #check if path exists
            if os.path.exists(i_path):
                print(f'Loading subject: {subdir}')
                data = scipy.io.loadmat(i_path)

                if  high_cut == 30.0:
                    print('Using 0-30 Hz band')

                    eeg = data['RAWfeature_0_30']
                    eeg = np.reshape(eeg, (eeg.shape[0], n_channels, 100))

                elif high_cut == 4.0:
                    print('Using 0-4 Hz band')

                    eeg = data['RAWfeature_0_4']
                    eeg = np.reshape(eeg, (eeg.shape[0], n_channels, 100))
                
                else:
                    raise ValueError(f"Unsupported high_cut value: {high_cut}. Expected 30 or 4.")

                labels = data['classID']
                labels = labels.flatten()
                print(f'EEG data shape: {eeg.shape}')
                print(f'Labels shape: {labels.shape}')

                jenna_dataset[subdir] = {'eeg': eeg, 'labels': labels}



    return jenna_dataset
