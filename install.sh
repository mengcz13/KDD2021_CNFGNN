conda install -y numpy scipy pandas jupyter matplotlib pyyaml scikit-learn tqdm networkx jupyterlab
conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install pytorch-lightning==0.9.0
pip install wandb
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-geometric
