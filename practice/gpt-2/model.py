# Imports
from dataclasses import dataclass
import glob
import argparse

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np



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

def block():
    pass

# Model Definition
class GPT2(nn.Module):
    def __init__(self, vocab_size=GPT2Config.n_vocab):
        super(GPT2, self).__init__()
        self.linear = nn.Linear(GPT2Config.n_embd, vocab_size)
        self.embed = nn.Embedding(vocab_size, GPT2Config.n_embd)

    def forward(self, x):
        #print("[gpt2] x shape:", x.shape)
        B, T = x.shape
        x = self.embed(x) # (B, T, n_embd)
        x = self.linear(x) # (B, T, 8)
        return x
    

# Training Loop
def train(args, model, data_loader):
    epochs = args.epochs
    lr = args.lr
    B = args.batch_size
    T = args.seq_length
    iterations = data_loader.ntok_total // (B * T * args.num_processes)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Training for %d epochs, %d iterations per epoch" % (epochs, iterations))
    for epoch in range(epochs):
        for it in range(iterations):
            x, y = data_loader.next_batch()
            #print("y", y.shape)

            pred = model(x)
            #print("pred", pred.shape)
            loss = ce_loss(pred.view(-1, GPT2Config.n_vocab), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # e.g., forward pass, loss computation, backward pass, optimizer step
            if it % 50 == 0:
                print(f"{epoch=}, {it=}, {loss.item()=:.4f}")


    
# Evaluation Metrics
# Save and Load Model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple transformer model.")
    parser.add_argument("--input_bin", type=str, default="data/tinyshakespeare/tiny_shakespeare_val.bin", help="Path to input binary data file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes for distributed data loading.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    loader = DistributedDataLoader(
        "/Users/calio/code/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin",
        B=args.batch_size,
        T=args.seq_length,
        process_rank=0,
        num_processes=1,
    )
    
    model = GPT2()
    train(args, model, loader)


if __name__ == "__main__":
    main()