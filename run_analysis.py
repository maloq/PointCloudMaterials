from src.training_methods.equivariant_autoencoder.predict_and_visualize import run_post_training_analysis

run_post_training_analysis(
    checkpoint_path="output/2026-01-28/18-41-16/EQ_AE_l120_N120_M80_chamfer+rdf_VN_REVNET_Anchor-epoch=148.ckpt",  # update path
    output_dir="output/eqae_analysis",  # where plots/metrics will go
    cuda_device=0,                      # GPU index (or leave as 0; uses CPU if no CUDA)
    cfg=None,                           # optional: pass a loaded Hydra cfg to skip reloading
)
