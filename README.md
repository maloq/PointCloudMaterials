# Pytorch Implementation of PointNet

# Create a new uv environment:

uv pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130

# install pytorch3d

conda install -c fvcore -c conda-forge fvcore -y
pip install iopath black usort flake8 flake8-bugbear flake8-comprehensions scikit-image matplotlib imageio plotly opencv-python
conda install pytorch3d -c pytorch3d-nightly -y

# install all other requirements
pip install -r requirements.txt