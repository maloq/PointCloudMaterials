from src.training_methods.equivariant_autoencoder.predict_and_visualize import run_post_training_analysis

run_post_training_analysis(
    checkpoint_path="output/2026-01-19/17-46-20/synth_SPD_VN_EQ_l96_P512_sinkhorn_64-epoch=43.ckpt",  # update path
    output_dir="output/eqae_analysis",  # where plots/metrics will go
    cuda_device=0,                      # GPU index (or leave as 0; uses CPU if no CUDA)
    cfg=None,                           # optional: pass a loaded Hydra cfg to skip reloading
    max_batches=32,                      # how many test batches to sample for visuals
)
