
import torch
import torch.nn as nn
from src.models.autoencoders.decoders.vn_decoders import VNRevnetAnchorDecoder

def test_decoder_config():
    print("Testing VNRevnetAnchorDecoder configuration...")
    
    # Configuration
    B = 2
    latent_size = 80
    
    # Instantiate Decoder with new parameters
    print("\nInstantiating VNRevnetAnchorDecoder with rich_invariants=True, rbf_dim=8...")
    decoder = VNRevnetAnchorDecoder(
        num_points=1024,
        latent_size=latent_size,
        rich_invariants=True,
        rbf_dim=8,
        rbf_cutoff=1.5
    )
    
    # Verify that the parameters reached the attention layer
    # VNRevnetAnchorDecoder -> blocks (list) -> VNAnchorTransformerBlock -> attn (VNChannelWiseSubtractionAttention)
    
    first_block = decoder.blocks[0]
    attn = first_block.attn
    
    print(f"Checking attention layer parameters...")
    print(f"attn.rich_invariants: {attn.rich_invariants}")
    print(f"attn.rbf_dim: {attn.rbf_dim}")
    
    assert attn.rich_invariants == True
    assert attn.rbf_dim == 8
    
    if attn.rbf is not None:
        print(f"attn.rbf.cutoff: {attn.rbf.cutoff}")
        assert attn.rbf.cutoff == 1.5
    else:
        print("Error: RBF module not initialized despite rbf_dim=8")
        assert False

    # Run a forward pass
    z = torch.randn(B, latent_size, 3)
    out = decoder(z)
    print(f"Forward pass successful. Output shape: {out.shape}")

    print("\nAll configuration tests passed!")

if __name__ == "__main__":
    try:
        test_decoder_config()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
