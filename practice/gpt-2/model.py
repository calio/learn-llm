# Imports
from dataclasses import dataclass
import glob
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import wandb
import tiktoken



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
# Data Loader
def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y
    

@dataclass
class GPT2Config:
    n_vocab: int = 50257
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

def create_positional_encoding2(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    
    # Create position indices [0, 1, 2, ..., seq_len-1]
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
    # Create dimension indices [0, 2, 4, ..., d_model-2]
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(torch.log(torch.tensor(10000.0)) / d_model))
    
    # Apply sin to even indices
    pe[:, 0::2] = torch.sin(position * div_term)
    
    # Apply cos to odd indices
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

def sinusoidal_positional_embedding(seq_len, d_model):
    # My implementation of sinusoidal positional embedding, needs validation
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(-1)
    div_term = torch.pow(torch.tensor(10000.0), torch.arange(0, d_model, 2).float() / d_model)

    print("position:", position.shape, "div_term", div_term.shape)
    pe[:, 0::2] = torch.sin(position.float() / div_term)
    pe[:, 1::2] = torch.cos(position.float() / div_term)
    return pe


def test_positional_encodings():
    """Test function to compare the two positional encoding implementations."""
    seq_len = 100
    d_model = 768
    
    # Generate embeddings with both functions
    pe1 = sinusoidal_positional_embedding(seq_len, d_model)
    pe2 = create_positional_encoding2(seq_len, d_model)
    
    # Check if they're close (allowing for small numerical differences)
    are_close = torch.allclose(pe1, pe2, atol=1e-6)
    
    # Calculate maximum absolute difference
    max_diff = torch.max(torch.abs(pe1 - pe2)).item()
    
    print(f"Positional encodings are close: {are_close}")
    print(f"Maximum absolute difference: {max_diff:.8f}")
    print(f"Shape of embeddings: {pe1.shape}")
    
    # Test with different dimensions
    for test_seq_len in [10, 50, 512]:
        for test_d_model in [64, 256, 768]:
            pe1_test = sinusoidal_positional_embedding(test_seq_len, test_d_model)
            pe2_test = create_positional_encoding2(test_seq_len, test_d_model)
            close = torch.allclose(pe1_test, pe2_test, atol=1e-6)
            max_diff_test = torch.max(torch.abs(pe1_test - pe2_test)).item()
            print(f"seq_len={test_seq_len}, d_model={test_d_model}: close={close}, max_diff={max_diff_test:.8f}")
    
    return are_close

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
    # - Use torch.matmul() or @ for matrix multiplication                     #
    # - For masking, use torch.where() with a large negative value (-1e9)    #
    # - Use F.softmax() or implement your own with torch.exp()               #
    # - For dropout, use F.dropout() if training                             #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    d_k = query.shape[-1]
    #print("Q", query.shape, "key.T", key.transpose(-1, -2).shape)
    x = torch.matmul(query, key.transpose(-1, -2))
    x = x / math.sqrt(d_k)

    if mask is not None:
        #print("mask", mask.shape)
        #print("x before mask", x.shape)
        # Apply the mask: set masked positions to a large negative value
        x = x.masked_fill(~mask, float('-inf'))
        #x = torch.where(mask, x, torch.tensor(-torch.inf, device=x.device))

    #print("x", x.shape)
    w = torch.softmax(x, -1)
    #print("w after softmax", w.shape)
    attention_weights = w
    output = torch.matmul(w, value)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention implementation matching gpt2_kapathy.py structure."""
    
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        """
        Initialize Multi-Head Attention.
        
        Args:
        - d_model: Model dimension
        - num_heads: Number of attention heads
        - dropout_p: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_p = dropout_p
        
        # Combined Q,K,V projection like gpt2_kapathy.py c_attn
        self.c_attn = nn.Linear(d_model, 3 * d_model)  # bias=True by default
        # Output projection like gpt2_kapathy.py c_proj  
        self.c_proj = nn.Linear(d_model, d_model)  # bias=True by default
        # Mark output projection for special scaling
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot uniform initialization."""
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, query, key, value, mask=None, training=True):
        """
        Forward pass for Multi-Head Attention matching gpt2_kapathy.py structure.
        
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
        B, T, C = query.size()
        
        # Combined Q,K,V projection like gpt2_kapathy.py
        qkv = self.c_attn(query)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply scaled dot-product attention
        x, attention_weights = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=self.dropout_p, training=training)
        
        # Re-assemble all head outputs side by side
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        output = self.c_proj(x)
        
        return output, attention_weights
    
class block(nn.Module):
    def __init__(self):
        super().__init__()
        # Register mask as a buffer so it moves with the model to the correct device
        self.register_buffer('mask', torch.tril(torch.ones(GPT2Config.n_ctx, GPT2Config.n_ctx, dtype=torch.bool)))

        self.mha = MultiHeadAttention(GPT2Config.n_embd, GPT2Config.n_head)
        self.norm1 = nn.LayerNorm(GPT2Config.n_embd)
        self.norm2 = nn.LayerNorm(GPT2Config.n_embd)
        self.ff_up = nn.Linear(GPT2Config.n_embd, 4 * GPT2Config.n_embd)
        self.ff_down = nn.Linear(4 * GPT2Config.n_embd, GPT2Config.n_embd)
        self.ff_down.LLMC_RESIDUAL_SCALE_FLAG = 1  # Mark for special scaling
        self.gelu = nn.GELU(approximate='tanh')


    def forward(self, x):
        # Create causal mask for the current sequence length
        B, T, C = x.shape
        mask = self.mask[:T, :T]  # Slice the mask to current sequence length
        
        x0 = x
        x = self.norm1(x)
        x = self.mha(x, x, x, mask=mask)[0]
        x = x + x0

        x0 = x
        x = self.norm2(x)
        x = self.ff_up(x)
        x = self.gelu(x)
        x = self.ff_down(x)
        x = x + x0
        return x


# Model Definition
class GPT2(nn.Module):
    def __init__(self, vocab_size=GPT2Config.n_vocab):
        super(GPT2, self).__init__()
        # Create layers to match gpt2_kapathy.py parameter structure
        self.embed = nn.Embedding(vocab_size, GPT2Config.n_embd)
        self.pos_embed = nn.Embedding(GPT2Config.n_ctx, GPT2Config.n_embd)
        self.blocks = nn.ModuleList([block() for _ in range(GPT2Config.n_layer)])
        self.ln = nn.LayerNorm(GPT2Config.n_embd)
        self.output_linear = nn.Linear(GPT2Config.n_embd, vocab_size, bias=False)  # No bias like gpt2_kapathy
        
        # Add weight tying like gpt2_kapathy.py 
        self.output_linear.LLMC_SKIP_INIT = 1  # Don't init this one, we will tie weights
        self.embed.weight = self.output_linear.weight  # Weight tying
        
        # Initialize weights with same method as gpt2_kapathy.py
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * GPT2Config.n_layer)
            # Skip initializing output_linear, which shares parameters with embed
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, x):
        #print("[gpt2] x shape:", x.shape)
        B, T = x.shape
        device = x.device  # Get device from input tensor
        x = self.embed(x) # (B, T, n_embd)
        pos = torch.arange(T, device=device)  # Create position tensor on same device
        x = x + self.pos_embed(pos)  # No need for unsqueeze(0), broadcasting handles it
        #x = self.linear(x) # (B, T, 8)
        for i in range(GPT2Config.n_layer):
            # Here you would typically apply transformer blocks, but for simplicity, we just pass through
            x = self.blocks[i](x)
        x = self.ln(x)
        x = self.output_linear(x) # (B, T, vocab_size)
        #x = F.softmax(x, dim=-1)
        return x

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= GPT2Config.n_ctx else idx[:, -GPT2Config.n_ctx:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    

def generate_sample_text(model, tokenizer, device, start_text="", max_new_tokens=32, temperature=1.0, top_k=40):
    """
    Generate sample text from the model following gpt2_kapathy.py pattern.
    
    Args:
        model: The GPT2 model
        tokenizer: tiktoken encoder 
        device: Device to run generation on
        start_text: Starting text (empty means start with EOT token)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
    
    Returns:
        Generated text string
    """
    model.eval()
    
    with torch.no_grad():
        # Start with end-of-text token like gpt2_kapathy.py
        if start_text:
            start_ids = tokenizer.encode(start_text)
        else:
            start_ids = [tokenizer.eot_token]  # Start with EOT token
        
        # Create input tensor
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        
        # Generate tokens
        generated = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        
        # Decode the generated sequence
        generated_text = tokenizer.decode(generated[0].tolist())
        
    model.train()
    return generated_text

def validate(args, model, data_loader):
    model.eval()
    device = get_device()

    with torch.no_grad():
        val_losses = []
        ce_loss = nn.CrossEntropyLoss()
        iterations = data_loader.ntok_total // (args.batch_size * args.seq_length * args.num_processes)
        for it in range(iterations):
            x, y = data_loader.next_batch()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = ce_loss(pred.view(-1, GPT2Config.n_vocab), y.view(-1))
            val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    model.train()
    return avg_val_loss

def generate(model, prompt_tokens, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    device = get_device()

    for _ in range(max_new_tokens):
        pass

    
        
# Training Loop
def train(args, model, data_loader):
    is_main_process = (not hasattr(args, 'local_rank') or args.local_rank == 0)

    epochs = args.epochs
    lr = args.lr
    B = args.batch_size
    T = args.seq_length
    iterations = data_loader.ntok_total // (B * T * args.num_processes)

    device = get_device()
    model = model.to(device)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_loader = DistributedDataLoader(args.input_val_bin, B=args.batch_size, T=args.seq_length, process_rank=0, num_processes=1)
    tokenizer = tiktoken.get_encoding("gpt2")

    print("device:", device)
    print("Training for %d epochs, %d iterations per epoch" % (epochs, iterations))
    
    global_step = 0
    for epoch in range(epochs):
        # Only show progress bar in main process
        if is_main_process:
            pbar = tqdm(range(iterations), desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        else:
            pbar = range(iterations)
        for it in pbar:
            x, y = data_loader.next_batch()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = ce_loss(pred.view(-1, GPT2Config.n_vocab), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update progress bar with loss information
            if is_main_process:
                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
            # Log training metrics to wandb
            if is_main_process and args.wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch": epoch + 1,
                    "train/learning_rate": lr,
                    "train/step": global_step
                }, step=global_step)
            # Run validation every eval_every iterations
            if global_step % args.eval_every == 0 and is_main_process:
                val_loss = validate(args, model, val_loader)
                if args.wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/epoch": epoch + 1
                    }, step=global_step)
            # Save checkpoint every save_every iterations
            if global_step % args.save_every == 0:
                import os
                os.makedirs(args.output_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_iter_{global_step}.pt")
                if is_main_process:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': global_step,
                        'epoch': epoch
                    }, checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")
            global_step += 1

            ## e.g., forward pass, loss computation, backward pass, optimizer step
            #if it % 50 == 0:
            #    print(f"{epoch=}, {it=}, {loss.item()=:.4f}")


        
        if is_main_process:
            generated_text = generate_sample_text(model, tokenizer, device, start_text="", max_new_tokens=256, temperature=1.0, top_k=40)
            print(f"Generated: {generated_text}")

            # Validation at the end of each epoch
            val_loss = validate(args, model, val_loader)
            
            # Log validation metrics to wandb
            if args.wandb:
                import os
                os.makedirs(args.output_dir, exist_ok=True)
                log_path = os.path.join(args.output_dir, "train.log")
                with open(log_path, "a") as f:
                    f.write(f"Epoch {epoch+1}, Step {global_step}, Val Loss: {val_loss}\n")
                wandb.log({
                    "val/loss": val_loss,
                    "val/epoch": epoch + 1
                }, step=global_step)


    
# Evaluation Metrics
# Save and Load Model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple transformer model.")
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_train.bin", help="Path to input binary data file.")
    parser.add_argument("--input_val_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="Path to validation binary data file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=GPT2Config.n_ctx, help="Sequence length.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes for distributed data loading.")
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (set by torch.distributed.launch)')
    
    # Wandb arguments
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="gpt-2", help="Wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (team/user name).")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name.")
    # Add eval_every argument
    parser.add_argument("--eval_every", type=int, default=100, help="How often (in iterations) to run validation.")
    parser.add_argument("--save_every", type=int, default=5000, help="How often (in iterations) to save a checkpoint.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to store checkpoints and logs.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    is_main_process = (not hasattr(args, 'local_rank') or args.local_rank == 0)

    # Wandb is enabled by default, unless --no_wandb is specified
    args.wandb = not args.no_wandb
    
    # Initialize wandb if enabled
    if is_main_process and args.wandb:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "batch_size": args.batch_size,
                "seq_length": args.seq_length,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "num_processes": args.num_processes,
                "vocab_size": GPT2Config.n_vocab,
                "n_ctx": GPT2Config.n_ctx,
                "n_embd": GPT2Config.n_embd,
                "n_head": GPT2Config.n_head,
                "n_layer": GPT2Config.n_layer,
            }
        )
        print(f"✅ Initialized wandb project: {args.wandb_project}")
        print(f"🔗 Run URL: {run.get_url()}")
    
    # Run positional encoding test
    #print("Testing positional encoding implementations...")
    #test_positional_encodings()
    #print("-" * 50)
    
    loader = DistributedDataLoader(
        args.input_bin,
        B=args.batch_size,
        T=args.seq_length,
        process_rank=0,
        num_processes=1,
    )
    
    model = GPT2()
    
    # Log model summary if wandb is enabled
    if args.wandb:
        wandb.watch(model, log="all", log_freq=10)
    
    if torch.cuda.device_count() > 1:
        import torch.distributed as dist
        import os
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend='nccl')
            device = torch.device('cuda', args.local_rank)
            model = nn.parallel.DistributedDataParallel(model.to(device), device_ids=[args.local_rank])
        else:
            device = torch.device('cuda')
            model = nn.DataParallel(model).to(device)
    else:
        device = get_device()
        model = model.to(device)
    
    train(args, model, loader)
    
    # Finish wandb run
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
