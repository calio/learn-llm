"""
Reference implementation of activation functions using PyTorch.
"""

import torch
import torch.nn.functional as F

class ActivationFunctions:
    """PyTorch reference implementations of activation functions."""
    
    @staticmethod
    def relu(x):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = F.relu(x_torch)
        return out.detach()
    
    @staticmethod
    def relu_grad(x):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = F.relu(x_torch)
        grad_out = torch.ones_like(out)
        out.backward(grad_out)
        return x_torch.grad
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = F.leaky_relu(x_torch, negative_slope=alpha)
        return out.detach()
    
    @staticmethod
    def leaky_relu_grad(x, alpha=0.01):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = F.leaky_relu(x_torch, negative_slope=alpha)
        grad_out = torch.ones_like(out)
        out.backward(grad_out)
        return x_torch.grad
    
    @staticmethod
    def sigmoid(x):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = torch.sigmoid(x_torch)
        return out.detach()
    
    @staticmethod
    def sigmoid_grad(x):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = torch.sigmoid(x_torch)
        grad_out = torch.ones_like(out)
        out.backward(grad_out)
        return x_torch.grad
    
    @staticmethod
    def tanh(x):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = torch.tanh(x_torch)
        return out.detach()
    
    @staticmethod
    def tanh_grad(x):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = torch.tanh(x_torch)
        grad_out = torch.ones_like(out)
        out.backward(grad_out)
        return x_torch.grad
    
    @staticmethod
    def swish(x):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = F.silu(x_torch)  # SiLU is the same as Swish
        return out.detach()
    
    @staticmethod
    def swish_grad(x):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = F.silu(x_torch)
        grad_out = torch.ones_like(out)
        out.backward(grad_out)
        return x_torch.grad
    
    @staticmethod
    def gelu(x, approximate=False):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        if approximate:
            out = F.gelu(x_torch, approximate='tanh')
        else:
            out = F.gelu(x_torch, approximate='none')
        return out.detach()
    
    @staticmethod
    def gelu_grad(x, approximate=False):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        if approximate:
            out = F.gelu(x_torch, approximate='tanh')
        else:
            out = F.gelu(x_torch, approximate='none')
        grad_out = torch.ones_like(out)
        out.backward(grad_out)
        return x_torch.grad
    
    @staticmethod
    def elu(x, alpha=1.0):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = F.elu(x_torch, alpha=alpha)
        return out.detach()
    
    @staticmethod
    def elu_grad(x, alpha=1.0):
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        out = F.elu(x_torch, alpha=alpha)
        grad_out = torch.ones_like(out)
        out.backward(grad_out)
        return x_torch.grad