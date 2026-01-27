
import torch
import torch.nn as nn
from src.models.autoencoders.decoders.vn_decoders import VNChannelWiseSubtractionAttention

def test_vn_attention():
    print("Testing VNChannelWiseSubtractionAttention upgrades...")
    
    # Configuration
    B, C, K = 2, 8, 4
    x = torch.randn(B, C, 3, K)
    pos = torch.randn(B, K, 3)
    
    # Case 1: Legacy mode (should match old behavior roughly in shape)
    print("\nCase 1: Legacy mode (rich_invariants=False, rbf_dim=0)")
    attn_legacy = VNChannelWiseSubtractionAttention(
        channels=C, 
        hidden=16, 
        use_pos=True, 
        rich_invariants=False, 
        rbf_dim=0
    )
    out_legacy = attn_legacy(x, pos)
    print(f"Output shape: {out_legacy.shape}")
    assert out_legacy.shape == (B, C, 3, K)
    
    # Case 2: Rich invariants only
    print("\nCase 2: Rich invariants (rich_invariants=True, rbf_dim=0)")
    attn_rich = VNChannelWiseSubtractionAttention(
        channels=C, 
        hidden=16, 
        use_pos=True, 
        rich_invariants=True, 
        rbf_dim=0
    )
    out_rich = attn_rich(x, pos)
    print(f"Output shape: {out_rich.shape}")
    assert out_rich.shape == (B, C, 3, K)
    
    # Case 3: RBF only
    print("\nCase 3: RBF only (rich_invariants=False, rbf_dim=8)")
    attn_rbf = VNChannelWiseSubtractionAttention(
        channels=C, 
        hidden=16, 
        use_pos=True, 
        rich_invariants=False, 
        rbf_dim=8
    )
    out_rbf = attn_rbf(x, pos)
    print(f"Output shape: {out_rbf.shape}")
    assert out_rbf.shape == (B, C, 3, K)
    
    # Case 4: Full upgrade
    print("\nCase 4: Full upgrade (rich_invariants=True, rbf_dim=8)")
    attn_full = VNChannelWiseSubtractionAttention(
        channels=C, 
        hidden=16, 
        use_pos=True, 
        rich_invariants=True, 
        rbf_dim=8
    )
    out_full = attn_full(x, pos)
    print(f"Output shape: {out_full.shape}")
    assert out_full.shape == (B, C, 3, K)

    # Check gradients
    loss = out_full.sum()
    loss.backward()
    print("Backward pass successful.")

    print("\nAll tests passed!")

if __name__ == "__main__":
    try:
        test_vn_attention()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
