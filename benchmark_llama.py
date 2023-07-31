import torch
import math
import pytest
from flashtriton.lama import ModelArgs, Attention

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

@pytest.mark.benchmark(group="Attention", timer=torch.cuda.synchronize)
def test_attention_forward(benchmark):
    # Run benchmark
    result = benchmark(model.forward, x, start_pos, freqs_cis, mask)

@pytest.mark.benchmark(group="Attention", timer=torch.cuda.synchronize)
def test_attention_forward_scaling(benchmark):
    # Modify sequence length and run benchmark
    args.max_seq_len = 16000
    x = torch.randn(args.max_batch_size, args.max_seq_len, args.dim).cuda()
    freqs_cis = torch.randn(args.max_batch_size, args.max_seq_len, args.dim).cuda()
    mask = torch.randn(args.max_batch_size, args.max_seq_len, args.dim).cuda()

    result = benchmark(model.forward, x, start_pos, freqs_cis, mask)
