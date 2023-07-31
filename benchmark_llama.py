import time
import torch
import pytest
from model import ModelArgs, Attention

# Model Arguments
args = ModelArgs(dim=512, n_heads=8, n_kv_heads=4, max_batch_size=64, max_seq_len=1024)

# Initialize model
model = Attention(args)
model.cuda()

# Generate some input data
x = torch.randn(args.max_batch_size, args.max_seq_len, args.dim).cuda()
start_pos = 0
freqs_cis = torch.randn(args.max_batch_size, args.max_seq_len, args.dim).cuda()
mask = torch.randn(args.max_batch_size, args.max_seq_len, args.dim).cuda()

def test_attention_forward():
    # Start timing
    start_time = time.time()

    # Run method
    model.forward(x, start_pos, freqs_cis, mask)

    # End timing
    end_time = time.time()

    # Print execution time
    print(f'Execution time for sequence length 1024: {end_time - start_time} milliseconds')

def test_attention_forward_scaling():
    # Modify sequence length and run benchmark
    args.max_seq_len = 16000
    x = torch.randn(args.max_batch_size, args.max_seq_len, args.dim).cuda()
    freqs_cis = torch.randn(args.max_batch_size, args.max_seq_len, args.dim).cuda()
    mask = torch.randn(args.max_batch_size, args.max_seq_len, args.dim).cuda()

    # Start timing
    start_time = time.time()

    # Run method
    model.forward(x, start_pos, freqs_cis, mask)

    # End timing
    end_time = time.time()

    # Print execution time
    print(f'Execution time for sequence length 16000: {end_time - start_time} milliseconds')
