import numpy as np
import torch

class StreamLayer(torch.nn.Module):
    def __init__(self, input_dimension=128, output_dimension=1):
        super(StreamLayer, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Linear(in_features=128, out_features=output_dimension)

    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        return output

# The NeuralNetwork class inherits the torch.nn.Module class, which represents a neural network.
class NeuralNetwork(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension, attention):
        # Call the initialisation function of the parent class.
        super(NeuralNetwork, self).__init__()
        # Define the network layers.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=64)
        #torch.nn.init.xavier_normal_(self.layer_1.weight)
        self.layer_2 = torch.nn.Linear(in_features=64, out_features=96)
        #torch.nn.init.xavier_normal_(self.layer_2.weight)
        self.layer_3 = torch.nn.Linear(in_features=96, out_features=64)
        #self.layer_4 = torch.nn.Linear(in_features=228, out_features=114)
        self.output_layer = torch.nn.Linear(in_features=64, out_features=output_dimension)
        #torch.nn.init.xavier_normal_(self.output_layer.weight)


    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.tanh(self.layer_1(input))
        layer_2_output = torch.tanh(self.layer_2(layer_1_output))
        layer_3_output = torch.tanh(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output



# The DQN class
class DQN:

    # class initialisation function.
    def __init__(self, input_dimension=1, output_dimension=24, gamma=0.9, lr=0.0001,
                 batch_size=32, attention=None):
        try:
            if torch.cuda.is_available():
                dev = 'cuda:0'
            elif torch.backends.mps.is_available():
                dev = 'mps'
            else:
                dev = 'cpu'

        except:
            if torch.cuda.is_available():
                dev = 'cuda:0'
            else:
                dev = 'cpu'
        self.device = torch.device(dev)
        self.input_dimension = input_dimension
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = NeuralNetwork(input_dimension=input_dimension, output_dimension=output_dimension, attention=attention).to(self.device)
        self.target_network = NeuralNetwork(input_dimension=input_dimension, output_dimension=output_dimension, attention=attention).to(self.device)
        self.update_target_network()
        # optimiser used when updating the Q-network.
        # learning rate determines how big each gradient step is during backpropagation.
        params = self.q_network.parameters()

        self.optimiser = torch.optim.Adam(params, lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size



    # Function to train the Q-network
    def train_q_network(self, minibatch, priority, entropy=None):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(minibatch, priority)
        if priority:
            updated_priorities = loss + 1e-5
            loss = loss.mean()
        if entropy:
            loss -= entropy * 0.01
        q_loss = loss

        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        if priority:
            return loss.item(), updated_priorities
        else:
            return loss.item()

    def compute_loss(self, minibatch, priority):
        # Calculate the loss for this transition.
        loss = self._calculate_loss(minibatch, priority)
        if priority:
            updated_priorities = loss + 1e-5
            loss = loss.mean()

        # Return the loss as a scalar
        if priority:
            return loss.item(), updated_priorities
        else:
            return loss.item()

    # Function to calculate the loss for a minibatch.
    def _calculate_loss(self, minibatch, priority):
        if priority:
            states, actions, rewards, next_states, buffer_indices, weights, dones = minibatch
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            states, actions, rewards, next_states, dones = minibatch
        #states = np.array(states)
        state_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        #next_states = np.array(next_states)
        next_state_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)
        # Calculate the predicted q-values for the current state
        state_q_values = self.q_network.forward(state_tensor)
        if state_q_values.shape[1] == 1:
            state_action_q_values = state_q_values
        else:
            state_action_q_values = state_q_values.gather(dim=1, index=action_tensor.unsqueeze(-1)).squeeze(-1)
        # Get the q-values for then next state
        next_state_q_values = self.target_network.forward(next_state_tensor).detach()  # Use .detach(), so that the target network is not updated
        # Get the maximum q-value
        next_state_max_q_values = next_state_q_values.max(1)[0]
        # Calculate the target q values
        target_state_action_q_values = reward_tensor + self.gamma * next_state_max_q_values * (1 - done_tensor)
        # Calculate the loss between the current estimates, and the target Q-values
        loss = torch.nn.MSELoss()(state_action_q_values, target_state_action_q_values)
        if priority:
            loss = loss * weight_tensor
        # Return the loss
        del state_tensor

        return loss

    def predict_q_values(self, state):
        if type(state) != torch.Tensor:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.clone().detach().unsqueeze(0).type(torch.float32).to(self.device)
        predicted_q_value_tensor = self.q_network.forward(state_tensor)

        return predicted_q_value_tensor.data.cpu().numpy()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
