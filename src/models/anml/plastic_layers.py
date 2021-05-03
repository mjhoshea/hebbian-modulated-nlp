import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F


class Plastic(nn.Module):
    '''Hebbian layer with weight decay (eta) as learnable parameter'''

    def __init__(self, in_features, out_features, activation=F.softmax):
        super().__init__()

        self.activation = activation

        # Regular weights
        self.w = Parameter(.01 * torch.randn(in_features, out_features), requires_grad=True)
        # Just a bias term
        self.b = Parameter(.01 * torch.randn(1), requires_grad=True)
        # Plasticity coefficients
        self.alpha = Parameter(.01 * torch.randn(in_features, out_features), requires_grad=True)
        # The weight decay term - "learning rate" of plasticity
        self.eta = Parameter(.01 * torch.randn(in_features, out_features), requires_grad=True)
        # Initialize hebbian trace
        self.trace = Variable(torch.zeros(in_features, out_features))

    def forward(self, x, is_training):
        if is_training:
            self.reset_trace()
        output = torch.zeros(x.shape[0], self.w.shape[-1])
        for i, x_in in enumerate(x):
            x_in = x_in.reshape(1, -1)
            x_out = self.activation(x_in.mm(self.w + torch.mul(self.alpha, self.trace)) + self.b)
            self.trace = (1 - self.eta) * self.trace + self.eta * torch.bmm(x_in.unsqueeze(2), x_out.unsqueeze(1))[0]
            self.trace = torch.clamp(self.trace, -1, 1)
            output[i] = x_out
        return output

    def reset_trace(self):
        self.trace = Variable(torch.zeros(self.w.shape))


class PlasticModulated(nn.Module):
    '''Hebbian layer with weight decay (eta) as input - nerunomodulation of plasticity'''

    def __init__(self, in_features, out_features, activation=F.softmax):
        super().__init__()

        self.activation = activation

        # Regular weights
        self.w = Parameter(.01 * torch.randn(in_features, out_features), requires_grad=True)
        # Just a bias term
        self.b = Parameter(.01 * torch.randn(1), requires_grad=True)
        # Plasticity coefficients
        self.alpha = Parameter(.01 * torch.randn(in_features, out_features), requires_grad=True)
        # Initialize hebbian trace
        self.trace = Variable(torch.zeros(in_features, out_features))

    def forward(self, x, eta, is_trainable):
        if is_trainable:
            self.reset_trace()

        output = torch.zeros(x.shape[0], self.w.shape[-1])
        for i, (x_in, eta_in) in enumerate(zip(x, eta)):
            eta_in = eta_in.reshape(self.w.shape)
            x_in = x_in.reshape(1, -1)
            x_out = self.activation(x_in.mm(self.w + torch.mul(self.alpha, self.trace)) + self.b)
            self.trace = (1 - eta_in) * self.trace + eta_in * torch.bmm(x_in.unsqueeze(2), x_out.unsqueeze(1))[0]
            self.trace = torch.clamp(self.trace, -1, 1)
            output[i] = x_out
        return output

    def reset_trace(self):
        self.trace = Variable(torch.zeros(self.w.shape))
