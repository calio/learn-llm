"""
Assignment 3: Attention Mechanisms
Your Name: _______________

Implement various attention mechanisms used in modern transformer architectures.
Fill in the TODO sections below.
"""

import numpy as np
import math

def scaled_dot_product_attention(query, key, value, mask=None, dropout_p=0.0, training=True):
    """
    Scaled Dot-Product Attention (SDPA) - the core attention mechanism.
    
    Inputs:
    - query: Query tensor of shape (..., seq_len_q, d_k)
    - key: Key tensor of shape (..., seq_len_k, d_k)
    - value: Value tensor of shape (..., seq_len_k, d_v)
    - mask: Optional attention mask of shape (..., seq_len_q, seq_len_k)
           where True/1 means positions to attend to, False/0 means mask out
    - dropout_p: Dropout probability for attention weights
    - training: Whether in training mode (affects dropout)
    
    Returns:
    - output: Attention output of shape (..., seq_len_q, d_v)
    - attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
    """
    output = None
    attention_weights = None
    
    #############################################################################
    # TODO: Implement Scaled Dot-Product Attention.                            #
    # Formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V                  #
    #                                                                           #
    # Steps:                                                                    #
    # 1. Compute attention scores: Q @ K^T                                     #
    # 2. Scale by sqrt(d_k) for numerical stability                           #
    # 3. Apply mask (set masked positions to large negative value)            #
    # 4. Apply softmax to get attention weights                               #
    # 5. Apply dropout if training                                             #
    # 6. Compute output: attention_weights @ V                                #
    #                                                                           #
    # Hints:                                                                    #
    # - Use np.matmul() or @ for matrix multiplication                        #
    # - For masking, use np.where() with a large negative value (-1e9)       #
    # - Implement softmax carefully to avoid overflow                         #
    # - For dropout, use np.random.binomial() if training                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return output, attention_weights

def softmax(x, axis=-1):
    """
    Numerically stable softmax implementation.
    
    Inputs:
    - x: Input array
    - axis: Axis along which to compute softmax
    
    Returns:
    - Softmax probabilities along specified axis
    """
    result = None
    
    #############################################################################
    # TODO: Implement numerically stable softmax.                              #
    # Use the identity: softmax(x) = softmax(x - max(x))                      #
    # This prevents overflow in the exponential function.                      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return result

class MultiHeadAttention:
    """Multi-Head Attention implementation."""
    
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        """
        Initialize Multi-Head Attention.
        
        Args:
        - d_model: Model dimension
        - num_heads: Number of attention heads
        - dropout_p: Dropout probability
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_p = dropout_p
        
        # Initialize weight matrices
        self.W_q = self._init_weights((d_model, d_model))
        self.W_k = self._init_weights((d_model, d_model))
        self.W_v = self._init_weights((d_model, d_model))
        self.W_o = self._init_weights((d_model, d_model))
        
    def _init_weights(self, shape):
        """Initialize weights with Xavier/Glorot uniform initialization."""
        limit = math.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, query, key, value, mask=None, training=True):
        """
        Forward pass for Multi-Head Attention.
        
        Inputs:
        - query: Query tensor of shape (batch_size, seq_len_q, d_model)
        - key: Key tensor of shape (batch_size, seq_len_k, d_model)
        - value: Value tensor of shape (batch_size, seq_len_k, d_model)
        - mask: Optional mask tensor
        - training: Training mode flag
        
        Returns:
        - output: Output tensor of shape (batch_size, seq_len_q, d_model)
        - attention_weights: Attention weights for visualization
        """
        output = None
        attention_weights = None
        
        #############################################################################
        # TODO: Implement Multi-Head Attention forward pass.                       #
        #                                                                           #
        # Steps:                                                                    #
        # 1. Apply linear transformations to get Q, K, V                          #
        # 2. Reshape and transpose to separate heads:                             #
        #    (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k) #
        # 3. Apply scaled dot-product attention for each head                      #
        # 4. Concatenate heads back together                                       #
        # 5. Apply output projection                                               #
        #                                                                           #
        # Hints:                                                                    #
        # - Use np.reshape() and np.transpose() for tensor manipulation          #
        # - Call scaled_dot_product_attention() for the core computation         #
        # - Remember to handle the mask shape for multiple heads                  #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return output, attention_weights

class MultiQueryAttention:
    """Multi-Query Attention (MQA) - shared key/value across heads."""
    
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        """
        Initialize Multi-Query Attention.
        
        Args:
        - d_model: Model dimension
        - num_heads: Number of query heads (key/value have only 1 head)
        - dropout_p: Dropout probability
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_p = dropout_p
        
        # Initialize weight matrices - note different shapes for K,V
        self.W_q = self._init_weights((d_model, d_model))
        self.W_k = self._init_weights((d_model, self.d_k))  # Single head for key
        self.W_v = self._init_weights((d_model, self.d_k))  # Single head for value
        self.W_o = self._init_weights((d_model, d_model))
        
    def _init_weights(self, shape):
        """Initialize weights with Xavier/Glorot uniform initialization."""
        limit = math.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, query, key, value, mask=None, training=True):
        """
        Forward pass for Multi-Query Attention.
        
        Inputs:
        - query: Query tensor of shape (batch_size, seq_len_q, d_model)
        - key: Key tensor of shape (batch_size, seq_len_k, d_model)
        - value: Value tensor of shape (batch_size, seq_len_k, d_model)
        - mask: Optional mask tensor
        - training: Training mode flag
        
        Returns:
        - output: Output tensor of shape (batch_size, seq_len_q, d_model)
        - attention_weights: Attention weights for visualization
        """
        output = None
        attention_weights = None
        
        #############################################################################
        # TODO: Implement Multi-Query Attention forward pass.                     #
        #                                                                           #
        # Key differences from MHA:                                                #
        # - Query still has num_heads, but Key/Value have only 1 head            #
        # - The single K,V head is broadcast/repeated across all query heads     #
        # - This reduces the KV cache size from h×d_k to 1×d_k per position     #
        #                                                                           #
        # Steps:                                                                    #
        # 1. Apply linear transformations: Q (multi-head), K,V (single head)     #
        # 2. Reshape Q to (batch_size, num_heads, seq_len, d_k)                  #
        # 3. K,V stay as (batch_size, 1, seq_len, d_k) or broadcast accordingly #
        # 4. Apply SDPA with broadcasting                                          #
        # 5. Concatenate and apply output projection                              #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return output, attention_weights

class GroupedQueryAttention:
    """Grouped-Query Attention (GQA) - groups of queries share key/value."""
    
    def __init__(self, d_model, num_query_heads, num_kv_heads, dropout_p=0.1):
        """
        Initialize Grouped-Query Attention.
        
        Args:
        - d_model: Model dimension
        - num_query_heads: Number of query heads
        - num_kv_heads: Number of key/value heads
        - dropout_p: Dropout probability
        """
        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_query_heads
        self.d_kv = d_model // num_query_heads  # KV head dimension
        self.dropout_p = dropout_p
        self.group_size = num_query_heads // num_kv_heads
        
        # Initialize weight matrices
        self.W_q = self._init_weights((d_model, d_model))
        self.W_k = self._init_weights((d_model, num_kv_heads * self.d_kv))
        self.W_v = self._init_weights((d_model, num_kv_heads * self.d_kv))
        self.W_o = self._init_weights((d_model, d_model))
        
    def _init_weights(self, shape):
        """Initialize weights with Xavier/Glorot uniform initialization."""
        limit = math.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, query, key, value, mask=None, training=True):
        """
        Forward pass for Grouped-Query Attention.
        
        Returns:
        - output: Output tensor of shape (batch_size, seq_len_q, d_model)
        - attention_weights: Attention weights for visualization
        """
        output = None
        attention_weights = None
        
        #############################################################################
        # TODO: Implement Grouped-Query Attention forward pass.                   #
        #                                                                           #
        # GQA interpolates between MHA (num_kv_heads = num_query_heads) and       #
        # MQA (num_kv_heads = 1). Each group of query heads shares a KV head.     #
        #                                                                           #
        # Steps:                                                                    #
        # 1. Apply linear transformations to get Q, K, V                          #
        # 2. Reshape Q to (batch_size, num_query_heads, seq_len, d_k)            #
        # 3. Reshape K,V to (batch_size, num_kv_heads, seq_len, d_kv)            #
        # 4. Repeat each KV head for its corresponding query group:               #
        #    K,V become (batch_size, num_query_heads, seq_len, d_kv)             #
        # 5. Apply SDPA and concatenate results                                    #
        #                                                                           #
        # Hint: Use np.repeat() to replicate KV heads for each group             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return output, attention_weights

class MultiHeadLatentAttention:
    """
    Multi-head Latent Attention (MLA) - DeepSeek-V2 style attention.
    Compresses KV representations into low-rank latent space.
    """
    
    def __init__(self, d_model, num_heads, latent_dim, dropout_p=0.1):
        """
        Initialize Multi-head Latent Attention.
        
        Args:
        - d_model: Model dimension
        - num_heads: Number of attention heads
        - latent_dim: Dimension of latent KV representations
        - dropout_p: Dropout probability
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.latent_dim = latent_dim
        self.dropout_p = dropout_p
        
        # Initialize weight matrices for MLA
        self.W_q = self._init_weights((d_model, d_model))
        
        # KV compression matrices
        self.W_kc = self._init_weights((d_model, latent_dim))  # Key compression
        self.W_vc = self._init_weights((d_model, latent_dim))  # Value compression
        
        # KV decompression matrices  
        self.W_ku = self._init_weights((latent_dim, num_heads * self.d_k))  # Key decompression
        self.W_vu = self._init_weights((latent_dim, num_heads * self.d_k))  # Value decompression
        
        self.W_o = self._init_weights((d_model, d_model))
        
    def _init_weights(self, shape):
        """Initialize weights with Xavier/Glorot uniform initialization."""
        limit = math.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, query, key, value, mask=None, training=True):
        """
        Forward pass for Multi-head Latent Attention.
        
        Returns:
        - output: Output tensor of shape (batch_size, seq_len_q, d_model)
        - attention_weights: Attention weights for visualization
        """
        output = None
        attention_weights = None
        
        #############################################################################
        # TODO: Implement Multi-head Latent Attention forward pass.               #
        #                                                                           #
        # MLA works by:                                                             #
        # 1. Compressing K,V into low-dimensional latent representations          #
        # 2. Decompressing these latents back to multi-head K,V                   #
        # 3. Applying standard multi-head attention                                #
        #                                                                           #
        # This reduces memory usage from O(seq_len * d_model * num_heads) to      #
        # O(seq_len * latent_dim) for the KV cache.                              #
        #                                                                           #
        # Steps:                                                                    #
        # 1. Apply query projection: Q = query @ W_q                              #
        # 2. Compress K,V: K_latent = key @ W_kc, V_latent = value @ W_vc        #
        # 3. Decompress K,V: K = K_latent @ W_ku, V = V_latent @ W_vu            #
        # 4. Reshape Q,K,V for multi-head attention                              #
        # 5. Apply SDPA and output projection                                      #
        #                                                                           #
        # The key insight is that the latent bottleneck reduces memory while      #
        # maintaining representational capacity through learned compression.       #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return output, attention_weights

def create_causal_mask(seq_len):
    """
    Create a causal (lower triangular) attention mask.
    
    Args:
    - seq_len: Sequence length
    
    Returns:
    - mask: Boolean mask of shape (seq_len, seq_len) where True means attend
    """
    mask = None
    
    #############################################################################
    # TODO: Create a causal mask for autoregressive attention.                 #
    # The mask should be lower triangular: position i can only attend to      #
    # positions j <= i (including itself).                                     #
    #                                                                           #
    # Hint: Use np.tril() (lower triangle) or manual indexing                 #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return mask

def create_padding_mask(seq_lengths, max_len):
    """
    Create padding mask for variable-length sequences.
    
    Args:
    - seq_lengths: Array of actual sequence lengths for each sample in batch
    - max_len: Maximum sequence length (padded length)
    
    Returns:
    - mask: Boolean mask of shape (batch_size, max_len) where True means valid token
    """
    mask = None
    
    #############################################################################
    # TODO: Create padding mask for batched sequences.                         #
    # For each sequence in the batch, positions beyond its actual length       #
    # should be masked out (False), while valid positions should be True.      #
    #                                                                           #
    # Hint: Use broadcasting with np.arange() and seq_lengths                  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return mask