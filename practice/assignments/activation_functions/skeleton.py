"""
Assignment 1: Activation Functions
Your Name: _______________

Implement forward and backward passes for common activation functions using PyTorch.
Fill in the TODO sections below.
"""

import torch
import torch.nn as nn
import math

def relu_forward(x):
    """
    ReLU activation function forward pass.
    
    Inputs:
    - x: Input tensor of any shape
    
    Returns:
    - out: Output tensor, same shape as x
    - cache: Values needed for backward pass
    """
    out = None
    cache = None
    
    #############################################################################
    # TODO: Implement the forward pass for ReLU activation.                    #
    # Store values needed for the backward pass in cache.                      #
    # Use PyTorch operations: torch.clamp(), torch.maximum(), or manual impl.  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return out, cache

def relu_backward(dout, cache):
    """
    ReLU activation function backward pass.
    
    Inputs:
    - dout: Upstream gradients, same shape as cache
    - cache: Values from forward pass
    
    Returns:
    - dx: Gradient with respect to x, same shape as dout
    """
    dx = None
    
    #############################################################################
    # TODO: Implement the backward pass for ReLU activation.                   #
    # Use the cached input values to determine gradient flow.                  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return dx

def leaky_relu_forward(x, alpha=0.01):
    """
    Leaky ReLU activation function forward pass.
    
    Inputs:
    - x: Input tensor of any shape
    - alpha: Leak parameter (slope for negative inputs)
    
    Returns:
    - out: Output tensor, same shape as x
    - cache: Values needed for backward pass
    """
    out = None
    cache = None
    
    #############################################################################
    # TODO: Implement the forward pass for Leaky ReLU activation.              #
    # Use torch.where() or manual implementation with conditions.              #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return out, cache

def leaky_relu_backward(dout, cache):
    """
    Leaky ReLU activation function backward pass.
    
    Inputs:
    - dout: Upstream gradients, same shape as cache
    - cache: Values from forward pass (should contain x and alpha)
    
    Returns:
    - dx: Gradient with respect to x, same shape as dout
    """
    dx = None
    
    #############################################################################
    # TODO: Implement the backward pass for Leaky ReLU activation.             #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return dx

def sigmoid_forward(x):
    """
    Sigmoid activation function forward pass.
    
    Inputs:
    - x: Input tensor of any shape
    
    Returns:
    - out: Output tensor, same shape as x
    - cache: Values needed for backward pass
    """
    out = None
    cache = None
    
    #############################################################################
    # TODO: Implement the forward pass for sigmoid activation.                 #
    # Be careful about numerical stability! Consider large positive/negative x. #
    # Use torch.sigmoid() or implement manually: 1 / (1 + torch.exp(-x))      #
    # For numerical stability, use torch.clamp() on inputs if needed.          #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return out, cache

def sigmoid_backward(dout, cache):
    """
    Sigmoid activation function backward pass.
    
    Inputs:
    - dout: Upstream gradients, same shape as cache
    - cache: Values from forward pass
    
    Returns:
    - dx: Gradient with respect to x, same shape as dout
    """
    dx = None
    
    #############################################################################
    # TODO: Implement the backward pass for sigmoid activation.                #
    # Use the property: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return dx

def tanh_forward(x):
    """
    Tanh activation function forward pass.
    
    Inputs:
    - x: Input tensor of any shape
    
    Returns:
    - out: Output tensor, same shape as x  
    - cache: Values needed for backward pass
    """
    out = None
    cache = None
    
    #############################################################################
    # TODO: Implement the forward pass for tanh activation.                    #
    # Use torch.tanh() or implement manually using torch.exp().               #
    # Consider numerical stability for extreme values.                         #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return out, cache

def tanh_backward(dout, cache):
    """
    Tanh activation function backward pass.
    
    Inputs:
    - dout: Upstream gradients, same shape as cache
    - cache: Values from forward pass
    
    Returns:
    - dx: Gradient with respect to x, same shape as dout
    """
    dx = None
    
    #############################################################################
    # TODO: Implement the backward pass for tanh activation.                   #
    # Use the property: d/dx tanh(x) = 1 - tanh^2(x)                          #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return dx

def swish_forward(x):
    """
    Swish/SiLU activation function forward pass: f(x) = x * sigmoid(x)
    
    Inputs:
    - x: Input tensor of any shape
    
    Returns:
    - out: Output tensor, same shape as x
    - cache: Values needed for backward pass
    """
    out = None
    cache = None
    
    #############################################################################
    # TODO: Implement the forward pass for Swish activation.                   #
    # Swish(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))                     #
    # You can use your sigmoid implementation or torch.sigmoid().              #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return out, cache

def swish_backward(dout, cache):
    """
    Swish/SiLU activation function backward pass.
    
    Inputs:
    - dout: Upstream gradients, same shape as cache
    - cache: Values from forward pass
    
    Returns:
    - dx: Gradient with respect to x, same shape as dout
    """
    dx = None
    
    #############################################################################
    # TODO: Implement the backward pass for Swish activation.                  #
    # d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return dx

def gelu_forward(x, approximate=False):
    """
    GELU activation function forward pass.
    
    Inputs:
    - x: Input tensor of any shape
    - approximate: If True, use the tanh approximation
                  If False, use exact GELU with error function
    
    Returns:
    - out: Output tensor, same shape as x
    - cache: Values needed for backward pass
    """
    out = None
    cache = None
    
    #############################################################################
    # TODO: Implement the forward pass for GELU activation.                    #
    # For exact: GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))        #
    # Use torch.erf() for the error function.                                  #
    # For approximate: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³))) #
    # You can also use torch.nn.functional.gelu() for reference.              #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return out, cache

def gelu_backward(dout, cache):
    """
    GELU activation function backward pass.
    
    Inputs:
    - dout: Upstream gradients, same shape as cache
    - cache: Values from forward pass
    
    Returns:
    - dx: Gradient with respect to x, same shape as dout
    """
    dx = None
    
    #############################################################################
    # TODO: Implement the backward pass for GELU activation.                   #
    # This is complex! For exact GELU:                                         #
    # d/dx GELU(x) = Φ(x) + x * φ(x) where φ(x) = (1/√(2π)) * exp(-x²/2)     #
    # For approximate, differentiate the tanh approximation.                   #
    # Use torch.exp(), torch.erf(), and mathematical constants.               #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return dx

def elu_forward(x, alpha=1.0):
    """
    ELU (Exponential Linear Unit) activation function forward pass.
    
    Inputs:
    - x: Input tensor of any shape
    - alpha: Scale parameter for negative inputs
    
    Returns:
    - out: Output tensor, same shape as x
    - cache: Values needed for backward pass
    """
    out = None
    cache = None
    
    #############################################################################
    # TODO: Implement the forward pass for ELU activation.                     #
    # ELU(x) = x if x > 0, else alpha * (exp(x) - 1)                          #
    # Use torch.where() to handle the conditional logic.                      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return out, cache

def elu_backward(dout, cache):
    """
    ELU activation function backward pass.
    
    Inputs:
    - dout: Upstream gradients, same shape as cache
    - cache: Values from forward pass
    
    Returns:
    - dx: Gradient with respect to x, same shape as dout
    """
    dx = None
    
    #############################################################################
    # TODO: Implement the backward pass for ELU activation.                    #
    # d/dx ELU(x) = 1 if x > 0, else alpha * exp(x)                          #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return dx