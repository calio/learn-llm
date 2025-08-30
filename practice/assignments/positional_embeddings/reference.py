"""
Reference implementation of positional embeddings using PyTorch.
"""

import torch
import torch.nn as nn
import math

class PositionalEmbeddings:
    """PyTorch reference implementations of positional embeddings."""
    
    @staticmethod
    def sinusoidal_positional_encoding(seq_len, d_model, base=10000):
        """Generate sinusoidal positional encodings."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(base) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    @staticmethod
    def learnable_positional_embedding(seq_len, d_model, init_std=0.02):
        """Generate learnable positional embeddings."""
        # Set seed for reproducible reference
        torch.manual_seed(42)
        pe = torch.normal(0, init_std, size=(seq_len, d_model))
        return pe
    
    @staticmethod
    def rope_positional_encoding(seq_len, d_model, base=10000):
        """Generate RoPE rotation matrices."""
        assert d_model % 2 == 0, "d_model must be even"
        
        # Create frequency tensor
        freqs = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        
        # Create position tensor
        t = torch.arange(seq_len).float()
        
        # Compute angles
        angles = torch.outer(t, freqs)  # (seq_len, d_model//2)
        
        # Expand to full dimension by repeating each frequency
        cos_cached = torch.cos(angles).repeat_interleave(2, dim=1)
        sin_cached = torch.sin(angles).repeat_interleave(2, dim=1)
        
        return cos_cached, sin_cached
    
    @staticmethod
    def apply_rope(x, cos_cached, sin_cached):
        """Apply RoPE rotation."""
        # Split into even and odd dimensions
        x_even = x[..., 0::2]  # x[..., [0, 2, 4, ...]]
        x_odd = x[..., 1::2]   # x[..., [1, 3, 5, ...]]
        
        cos_even = cos_cached[..., 0::2]
        cos_odd = cos_cached[..., 1::2] 
        sin_even = sin_cached[..., 0::2]
        sin_odd = sin_cached[..., 1::2]
        
        # Apply rotation
        x_even_rot = x_even * cos_even - x_odd * sin_even
        x_odd_rot = x_even * sin_odd + x_odd * cos_odd
        
        # Interleave back
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2)
        
        return x_rot
    
    @staticmethod  
    def relative_position_encoding(seq_len, max_distance=32):
        """Generate relative position matrix."""
        positions = torch.arange(seq_len)
        relative_positions = positions[:, None] - positions[None, :]
        
        # Clip to max distance
        relative_positions = torch.clamp(relative_positions, -max_distance, max_distance)
        
        # Shift to make indices positive (for embedding lookup)
        relative_positions = relative_positions + max_distance
        
        return relative_positions
    
    @staticmethod
    def create_relative_position_embeddings(max_distance, d_model, init_std=0.02):
        """Create relative position embedding table."""
        torch.manual_seed(42)  # For reproducible reference
        num_embeddings = 2 * max_distance + 1
        embeddings = torch.normal(0, init_std, size=(num_embeddings, d_model))
        return embeddings
    
    @staticmethod
    def lookup_relative_embeddings(relative_positions, embedding_table):
        """Look up relative embeddings."""
        relative_positions = relative_positions.long()
        embeddings = embedding_table[relative_positions]
        return embeddings
    
    @staticmethod
    def alibi_slopes(num_heads):
        """Generate ALiBi slopes."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(get_slopes_power_of_2(2*closest_power_of_2)[0::2][:num_heads-closest_power_of_2])
        
        return torch.tensor(slopes)
    
    @staticmethod
    def create_alibi_bias(seq_len, slopes):
        """Create ALiBi bias matrix."""
        num_heads = len(slopes)
        
        # Create distance matrix
        arange = torch.arange(seq_len)
        distances = torch.abs(arange[:, None] - arange[None, :])
        
        # Apply slopes
        bias = slopes.view(num_heads, 1, 1) * distances.unsqueeze(0)
        
        # Make negative (typical convention)
        bias = -bias
        
        return bias