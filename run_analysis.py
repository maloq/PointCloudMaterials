from src.training_methods.equivariant_autoencoder.predict_and_visualize import run_post_training_analysis

run_post_training_analysis(
    checkpoint_path="output/2026-01-26/14-52-10/EQ_AE_l80_P80_chamfer_VN_REVNET_Anchor-epoch=05.ckpt",  # update path
    output_dir="output/eqae_analysis",  # where plots/metrics will go
    cuda_device=0,                      # GPU index (or leave as 0; uses CPU if no CUDA)
    cfg=None,                           # optional: pass a loaded Hydra cfg to skip reloading
)
