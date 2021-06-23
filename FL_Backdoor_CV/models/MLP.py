import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_MLP(nn.Module):
    """
        global batch_size = 100
    """
    def __init__(self, num_classes):
        super(MNIST_MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(28*28, 500))
        self.layers.append(nn.Linear(500, 500))
        self.layers.append(nn.Linear(500, num_classes))
        # self.fc1 = nn.Linear(28*28, 500)
        # self.fc2 = nn.Linear(500, 500)
        # self.fc3 = nn.Linear(500, 10)

    def forward(self, x): # x: (batch, )
        # x = x.view(-1, 28*28)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x
        x = x.view(-1, 28 * 28)
        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        x = self.layers[2](x)
        return x

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weight)
        return weights

    def get_gradients(self):
        gradients = []
        for layer in self.layers:
            gradients.append(layer.weight.grad)

        return gradients

    def assign_gradients(self, gradients):
        for idx, layer in enumerate(self.layers):
            layer.weight.grad.data = gradients[idx]

    def update_weights(self, gradients, lr):
        for idx, layer in enumerate(self.layers):
            layer.weight.data -= lr * gradients[idx].data

    def initialize_new_grads(self):
        init_grads = []
        for layer in self.layers:
            init_grads.append(torch.zeros_like(layer.weight))
        return init_grads

