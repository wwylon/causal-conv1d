import torch
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

def test_basic():
    device = "cuda"
    itype = torch.float16
    batch, dim, seqlen, width = 2, 64, 128, 4
    
    print(f"Testing causal_conv1d_fn with {itype}...")
    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32).requires_grad_()
    bias = torch.randn(dim, device=device, dtype=torch.float32).requires_grad_()
    
    out = causal_conv1d_fn(x, weight, bias, activation="silu")
    print(f"Forward pass successful. Output shape: {out.shape}")
    
    grad = torch.randn_like(out)
    out.backward(grad)
    print("Backward pass successful.")
    
    print(f"Testing causal_conv1d_update with {itype}...")
    conv_state = torch.randn(batch, width, dim, device=device, dtype=itype).transpose(1, 2)
    x_update = torch.randn(batch, 1, dim, device=device, dtype=itype).transpose(1, 2)
    
    out_update = causal_conv1d_update(x_update, conv_state, weight, bias, activation="silu")
    print(f"Update successful. Output shape: {out_update.shape}")
    
    print("\nAll basic tests passed!")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping tests.")
    else:
        try:
            test_basic()
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
