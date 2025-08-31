"""
Reference implementation of attention mechanisms using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AttentionReference:
    """PyTorch reference implementations of attention mechanisms."""
    
    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask=None, dropout_p=0.0, training=True):
        """Reference SDPA implementation."""
        query = torch.tensor(query, dtype=torch.float32)
        key = torch.tensor(key, dtype=torch.float32)
        value = torch.tensor(value, dtype=torch.float32)
        
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool)
        
        # Use PyTorch's efficient SDPA if available
        if hasattr(F, 'scaled_dot_product_attention'):
            # Convert mask format if needed
            attn_mask = None
            if mask is not None:
                attn_mask = ~mask  # PyTorch uses True for masked positions
                attn_mask = attn_mask.float() * -1e9
            
            output = F.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=attn_mask,
                dropout_p=dropout_p if training else 0.0,
                is_causal=False
            )
            
            # Compute attention weights for comparison
            d_k = query.size(-1)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            if attn_mask is not None:
                scores = scores + attn_mask
            attention_weights = F.softmax(scores, dim=-1)
            
            return output.detach(), attention_weights.detach()
        else:
            # Manual implementation
            d_k = query.size(-1)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            
            if mask is not None:
                scores = scores.masked_fill(~mask, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            
            if training and dropout_p > 0:
                attention_weights = F.dropout(attention_weights, p=dropout_p, training=training)
            
            output = torch.matmul(attention_weights, value)
            
            return output.detach(), attention_weights.detach()

class MultiHeadAttentionReference(nn.Module):
    """PyTorch reference Multi-Head Attention."""
    
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout_p)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
    
    def forward(self, query, key, value, mask=None, training=True):
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)
        
        # Linear transformations
        Q = self.W_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.W_v(value)  # (batch_size, seq_len_k, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = AttentionReference.scaled_dot_product_attention(
            Q.detach().numpy(), K.detach().numpy(), V.detach().numpy(), mask, training=training
        )
        
        attention_output = attention_output.clone().detach()
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Output projection
        output = self.W_o(attention_output)
        
        return output.detach(), attention_weights

class MultiQueryAttentionReference(nn.Module):
    """PyTorch reference Multi-Query Attention."""
    
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)  # Single head
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)  # Single head
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
    
    def forward(self, query, key, value, mask=None, training=True):
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)
        
        # Linear transformations
        Q = self.W_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)    # (batch_size, seq_len_k, d_k)
        V = self.W_v(value)  # (batch_size, seq_len_k, d_k)
        
        # Reshape Q for multi-head
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        
        # Expand K,V to match number of heads
        K = K.unsqueeze(1).expand(batch_size, self.num_heads, seq_len_k, self.d_k)
        V = V.unsqueeze(1).expand(batch_size, self.num_heads, seq_len_k, self.d_k)
        
        # Apply attention
        attention_output, attention_weights = AttentionReference.scaled_dot_product_attention(
            Q.detach().numpy(), K.detach().numpy(), V.detach().numpy(), mask, training=training
        )
        
        attention_output = attention_output.clone().detach()
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Output projection
        output = self.W_o(attention_output)
        
        return output.detach(), attention_weights

class GroupedQueryAttentionReference(nn.Module):
    """PyTorch reference Grouped-Query Attention."""
    
    def __init__(self, d_model, num_query_heads, num_kv_heads, dropout_p=0.1):
        super().__init__()
        assert d_model % num_query_heads == 0
        assert num_query_heads % num_kv_heads == 0
        
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_query_heads
        self.d_kv = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_kv, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_kv, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
    
    def forward(self, query, key, value, mask=None, training=True):
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)
        
        # Linear transformations
        Q = self.W_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)    # (batch_size, seq_len_k, num_kv_heads * d_kv)
        V = self.W_v(value)  # (batch_size, seq_len_k, num_kv_heads * d_kv)
        
        # Reshape for attention
        Q = Q.view(batch_size, seq_len_q, self.num_query_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_kv_heads, self.d_kv).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_kv_heads, self.d_kv).transpose(1, 2)
        
        # Repeat K,V for each group
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)
        
        # Apply attention
        attention_output, attention_weights = AttentionReference.scaled_dot_product_attention(
            Q.detach().numpy(), K.detach().numpy(), V.detach().numpy(), mask, training=training
        )
        
        attention_output = attention_output.clone().detach()
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Output projection
        output = self.W_o(attention_output)
        
        return output.detach(), attention_weights

class MultiHeadLatentAttentionReference(nn.Module):
    """PyTorch reference Multi-head Latent Attention."""
    
    def __init__(self, d_model, num_heads, latent_dim, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.latent_dim = latent_dim
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        
        # KV compression/decompression
        self.W_kc = nn.Linear(d_model, latent_dim, bias=False)
        self.W_vc = nn.Linear(d_model, latent_dim, bias=False)
        self.W_ku = nn.Linear(latent_dim, num_heads * self.d_k, bias=False)
        self.W_vu = nn.Linear(latent_dim, num_heads * self.d_k, bias=False)
        
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Initialize weights
        for layer in [self.W_q, self.W_kc, self.W_vc, self.W_ku, self.W_vu, self.W_o]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, query, key, value, mask=None, training=True):
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)
        
        # Query projection
        Q = self.W_q(query)
        
        # KV compression
        K_latent = self.W_kc(key)    # (batch_size, seq_len_k, latent_dim)
        V_latent = self.W_vc(value)  # (batch_size, seq_len_k, latent_dim)
        
        # KV decompression
        K = self.W_ku(K_latent)  # (batch_size, seq_len_k, num_heads * d_k)
        V = self.W_vu(V_latent)  # (batch_size, seq_len_k, num_heads * d_k)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = AttentionReference.scaled_dot_product_attention(
            Q.detach().numpy(), K.detach().numpy(), V.detach().numpy(), mask, training=training
        )
        
        attention_output = attention_output.clone().detach()
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Output projection
        output = self.W_o(attention_output)
        
        return output.detach(), attention_weights

class UtilityFunctions:
    """Utility functions for attention mechanisms."""
    
    @staticmethod
    def create_causal_mask(seq_len):
        """Create causal attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        return mask
    
    @staticmethod
    def create_padding_mask(seq_lengths, max_len):
        """Create padding mask for variable sequences."""
        batch_size = len(seq_lengths)
        mask = torch.arange(max_len)[None, :] < torch.tensor(seq_lengths)[:, None]
        return mask