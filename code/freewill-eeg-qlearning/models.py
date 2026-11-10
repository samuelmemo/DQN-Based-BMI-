import torch
from torch import nn
import torch.optim as optim
import random
import math
import numpy as np
from collections import deque
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNLSTM_EEG(nn.Module):
    """
    Input:  x of shape (N, C, T)  where:
            N = batch size
            C = EEG channels
            T = time points

    Output: logits of shape (N, n_classes)
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 4,
        fs: int = 200,
        conv1_out: int = 64,
        conv2_out: int = 128,
        pool_type: str = "avg",   # "max", "avg", or "both" (concat)
        pool_kernel: int = 4,
        pool_stride: int = 4,
        dropout_p: float = 0.1,
        lstm_hidden: int = 128,
        fc_hidden: int = 128,
    ):
        super().__init__()

      
        k1 = fs//2
        k2 = fs//4

        self.k1, self.k2 = k1, k2
        self.pool_type = pool_type.lower()

        # ---- 1D CNN feature extractor over time ----
        # Note: Conv1d expects (N, C_in, T).
        # Order: Conv → BatchNorm → Activation (bias=False since BatchNorm has bias)
        self.conv1 = nn.Conv1d(n_channels, conv1_out, kernel_size=k1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.act1 = nn.PReLU(num_parameters=conv1_out)

        # output shape after conv1: (N, conv1_out, T)
    
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=k2, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.act2 = nn.PReLU(num_parameters=conv2_out)

        # Pooling (choose one or both)
        self.maxpool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)
        self.avgpool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride)

        # Drop entire feature maps (channels) consistently across time.
        self.spatial_dropout = nn.Dropout1d(p=dropout_p)

        
        lstm_input_size = conv2_out 

        # ---- LSTM over time ----
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,  # (N, L, F)
            bidirectional=False,
        )

        # ---- Fully connected head ----
        self.fc1 = nn.Linear(lstm_hidden, fc_hidden)
        self.fc_out = nn.Linear(fc_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, T)
        returns logits: (N, n_classes)
        """
        # CNN
        x = self.act1(self.bn1(self.conv1(x)))  # (N, conv1_out, T)
        x = self.act2(self.bn2(self.conv2(x)))  # (N, conv2_out, T)

        # Pool
        if self.pool_type == "max":
            x = self.maxpool(x)
        elif self.pool_type == "avg":
            x = self.avgpool(x)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        x = self.spatial_dropout(x)  # (N, F, T)

        # LSTM expects (N, T, F)
        x = x.transpose(1, 2)  # (N, T, F)
        x, _ = self.lstm(x)    # (N, T, lstm_hidden)

        # GlobalMaxPool over time dimension T
        x = torch.max(x, dim=1).values  # (N, lstm_hidden)

        # FC head
        x = F.relu(self.fc1(x))
        out = self.fc_out(x)         # (N, n_classes)
        return out


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=31, Samples=125, dropoutRate=0.5, 
                kernLength=125, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding = 'same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)

        # Block 2
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding = 'same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)

        # Classification Layer
        self.flatten = nn.Flatten()
        
        # Calculate input size for the linear layer
        # Input: (1, Chans, Samples)
        # Conv1: (F1, Chans, Samples)
        # Depthwise: (F1*D, 1, Samples)
        # Pool1: (F1*D, 1, Samples // 4)
        # Separable: (F2, 1, Samples // 4)
        # Pool2: (F2, 1, Samples // 32)
        
        out_samples = Samples // 32
        self.fc = nn.Linear(F2 * out_samples, nb_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)
    



class DQNAgent:
    def __init__(self, nb_classes, chans, samples, kernLength,model_name, lr=1e-3, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=25, buffer_size=10000, batch_size=64):
        self.nb_classes = nb_classes
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.steps_done = 0

        if model_name == 'EEGNet':
            print("Using EEGNet model")
            self.policy_net = EEGNet(nb_classes, Chans=chans, Samples=samples, kernLength=kernLength).to(device)
            self.target_net = EEGNet(nb_classes, Chans=chans, Samples=samples, kernLength=kernLength).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        elif model_name == 'CNNLSTM_EEG':
            print("Using CNNLSTM_EEG model")
            self.policy_net = CNNLSTM_EEG(n_channels=chans, n_classes=nb_classes, fs=kernLength*2).to(device)
            self.target_net = CNNLSTM_EEG(n_channels=chans, n_classes=nb_classes, fs=kernLength*2).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """
        Select action using epsilon-greedy policy with exploration rate epsilon = 0.01.
        """
        epsilon = 0.01  # Fixed exploration rate
        sample = random.random()
        
        # Explore: select random action with probability epsilon
        if sample < epsilon:
            return random.randrange(self.nb_classes)
        # Exploit: select action with highest Q-value
        else:
            self.policy_net.eval()
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state)
                self.policy_net.train()
                return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # Ensure training mode for RNNs (cudnn requires train mode for backward)
        self.policy_net.train()

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).unsqueeze(1).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        q_values = self.policy_net(state).gather(1, action)
        # Detach target Q-values to prevent backprop through target network (cudnn RNN requires train mode otherwise)
        next_q_values = self.target_net(next_state).max(1)[0].unsqueeze(1).detach()
        expected_q_values = reward + (self.gamma * next_q_values * (1 - done))

        loss = self.loss_fn(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    #def decay_epsilon(self):
    #    self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
