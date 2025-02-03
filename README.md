# Pytorch Implementation of PointNet

# Create a new environment

conda create -n pointnet python=3.12 -y
conda activate pointnet
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# install pytorch3d

conda install -c fvcore -c conda-forge fvcore -y
pip install iopath black usort flake8 flake8-bugbear flake8-comprehensions scikit-image matplotlib imageio plotly opencv-python
conda install pytorch3d -c pytorch3d-nightly -y

# install all other requirements
pip install -r requirements.txt