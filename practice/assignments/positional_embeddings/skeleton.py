"""
Assignment 2: Positional Embeddings
Your Name: _______________

Implement various positional encoding schemes used in transformer architectures.
Fill in the TODO sections below.
"""

import numpy as np
import math

def sinusoidal_positional_encoding(seq_len, d_model, base=10000):
    """
    Generate sinusoidal positional encodings as used in the original Transformer paper.
    
    Inputs:
    - seq_len: Maximum sequence length
    - d_model: Model dimension (embedding size)
    - base: Base for the exponential (default 10000)
    
    Returns:
    - pos_encoding: Array of shape (seq_len, d_model) containing positional encodings
    """
    pos_encoding = None
    
    #############################################################################
    # TODO: Implement sinusoidal positional encoding.                          #
    # For position pos and dimension i:                                        #
    # PE(pos, 2i) = sin(pos / base^(2i/d_model))                              #
    # PE(pos, 2i+1) = cos(pos / base^(2i/d_model))                            #
    #                                                                           #
    # Hints:                                                                    #
    # - Create arrays for positions [0, 1, ..., seq_len-1]                    #
    # - Create arrays for dimensions [0, 1, ..., d_model-1]                   #
    # - Use broadcasting to compute all combinations efficiently               #
    # - Be careful about even/odd indexing                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return pos_encoding

def learnable_positional_embedding(seq_len, d_model, init_std=0.02):
    """
    Generate learnable positional embeddings (randomly initialized).
    
    Inputs:
    - seq_len: Maximum sequence length
    - d_model: Model dimension (embedding size)  
    - init_std: Standard deviation for initialization
    
    Returns:
    - pos_embedding: Array of shape (seq_len, d_model) randomly initialized
    """
    pos_embedding = None
    
    #############################################################################
    # TODO: Generate randomly initialized learnable positional embeddings.     #
    # Initialize from a normal distribution with mean=0, std=init_std          #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return pos_embedding

def rope_positional_encoding(seq_len, d_model, base=10000):
    """
    Generate Rotary Position Embedding (RoPE) rotation matrices.
    
    Inputs:
    - seq_len: Maximum sequence length
    - d_model: Model dimension (must be even for RoPE)
    - base: Base for the exponential
    
    Returns:
    - cos_cached: Cosine values for rotation, shape (seq_len, d_model)
    - sin_cached: Sine values for rotation, shape (seq_len, d_model)
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for RoPE")
    
    cos_cached = None
    sin_cached = None
    
    #############################################################################
    # TODO: Implement RoPE positional encoding.                                #
    # RoPE uses rotation matrices parameterized by position and frequency.     #
    #                                                                           #
    # For each pair of dimensions (2i, 2i+1), we have:                        #
    # theta_i = base^(-2i/d_model)                                             #
    # For position m: angle = m * theta_i                                      #
    #                                                                           #
    # The rotation matrix becomes:                                              #
    # [cos(angle), -sin(angle)]                                                #
    # [sin(angle),  cos(angle)]                                                #
    #                                                                           #
    # We precompute cos and sin values for all positions and dimensions.       #
    #                                                                           #
    # Hints:                                                                    #
    # - Create frequency array: base^(-2*i/d_model) for i in [0, d_model/2)   #
    # - Each frequency is used for 2 consecutive dimensions                    #
    # - Use broadcasting: positions[:, None] * freqs[None, :]                  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return cos_cached, sin_cached

def apply_rope(x, cos_cached, sin_cached):
    """
    Apply RoPE rotation to input embeddings.
    
    Inputs:
    - x: Input embeddings of shape (..., seq_len, d_model)
    - cos_cached: Precomputed cosine values, shape (seq_len, d_model)
    - sin_cached: Precomputed sine values, shape (seq_len, d_model)
    
    Returns:
    - x_rotated: Rotated embeddings, same shape as x
    """
    x_rotated = None
    
    #############################################################################
    # TODO: Apply RoPE rotation to the input embeddings.                       #
    # For each pair of dimensions (2i, 2i+1):                                  #
    # x_rotated[..., 2i] = x[..., 2i] * cos - x[..., 2i+1] * sin             #
    # x_rotated[..., 2i+1] = x[..., 2i] * sin + x[..., 2i+1] * cos           #
    #                                                                           #
    # Hints:                                                                    #
    # - Split x into even and odd indexed dimensions                           #
    # - Apply rotation formula element-wise                                     #
    # - Recombine the rotated dimensions                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return x_rotated

def relative_position_encoding(seq_len, max_distance=32):
    """
    Generate T5-style relative position encodings.
    
    Inputs:
    - seq_len: Sequence length
    - max_distance: Maximum relative distance to consider
    
    Returns:
    - relative_positions: Array of shape (seq_len, seq_len) containing relative position indices
    """
    relative_positions = None
    
    #############################################################################
    # TODO: Generate relative position matrix as used in T5.                   #
    # For positions i and j, the relative position is (i - j).                #
    # Clip the relative positions to [-max_distance, max_distance].            #
    #                                                                           #
    # The output should contain indices that can be used to lookup embeddings  #
    # from a relative position embedding table.                                #
    #                                                                           #
    # Hints:                                                                    #
    # - Create position indices [0, 1, ..., seq_len-1]                        #
    # - Use broadcasting to compute all pairwise differences                   #
    # - Apply clipping to limit the range                                      #
    # - Shift indices to be positive for embedding lookup                      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return relative_positions

def create_relative_position_embeddings(max_distance, d_model, init_std=0.02):
    """
    Create learnable relative position embedding table.
    
    Inputs:
    - max_distance: Maximum relative distance
    - d_model: Embedding dimension
    - init_std: Standard deviation for initialization
    
    Returns:
    - embeddings: Array of shape (2*max_distance + 1, d_model) containing embeddings
                 Index 0 corresponds to relative position -max_distance
                 Index max_distance corresponds to relative position 0
                 Index 2*max_distance corresponds to relative position +max_distance
    """
    embeddings = None
    
    #############################################################################
    # TODO: Create embedding table for relative positions.                     #
    # The table should contain embeddings for positions from -max_distance     #
    # to +max_distance (inclusive), so total size is 2*max_distance + 1.      #
    # Initialize from normal distribution with given std.                      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return embeddings

def lookup_relative_embeddings(relative_positions, embedding_table):
    """
    Look up relative position embeddings using the position indices.
    
    Inputs:
    - relative_positions: Array of shape (seq_len, seq_len) with position indices
    - embedding_table: Array of shape (2*max_distance + 1, d_model)
    
    Returns:
    - embeddings: Array of shape (seq_len, seq_len, d_model) containing looked-up embeddings
    """
    embeddings = None
    
    #############################################################################
    # TODO: Look up embeddings from the table using relative position indices. #
    # Use the relative_positions array to index into the embedding_table.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return embeddings