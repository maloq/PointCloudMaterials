from src.training_methods.equivariant_autoencoder.predict_and_visualize import run_post_training_analysis

run_post_training_analysis(
    checkpoint_path="output/2026-01-23/17-57-38/EQ_AE_l720_P512_chamfer_VN_Equivariant-epoch=67.ckpt",  # update path
    output_dir="output/eqae_analysis",  # where plots/metrics will go
    cuda_device=0,                      # GPU index (or leave as 0; uses CPU if no CUDA)
    cfg=None,                           # optional: pass a loaded Hydra cfg to skip reloading
)
