import time
import torch
import pytest

from flashtriton.flash_torch import FlashAttention

# Model Arguments
args = {
    "dim": 512, 
    "heads": 8, 
    "dim_head": 64, 
    "causal": False, 
    "q_bucket_size": 512, 
    "k_bucket_size": 1024, 
    "parallel": False, 
    "mixed_precision": False
}

# Initialize model
model = FlashAttention(**args)
model.cuda()

# Generate some input data
x = torch.randn(64, 1024, args['dim']).cuda()

def test_flash_attention_forward():
    # Start timing
    start_time = time.time()

    # Run method
    model(x)

    # End timing
    end_time = time.time()

    # Print execution time
    print(f'Execution time for sequence length 1024: {end_time - start_time} milliseconds')

def test_flash_attention_forward_scaling():
    # Modify sequence length and run benchmark
    x = torch.randn(64, 16000, args['dim']).cuda()

    # Start timing
    start_time = time.time()

    # Run method
    model(x)

    # End timing
    end_time = time.time()

    # Print execution time
    print(f'Execution time for sequence length 16000: {end_time - start_time} milliseconds')

# Run tests
if __name__ == "__main__":
    test_flash_attention_forward()
    test_flash_attention_forward_scaling()
