#!/bin/bash
# install python dependencies

source activate habitat

# remove old package
pip uninstall -y torch
pip uninstall -y tensorflow tb-nightly tensorboard tensorboard-data-server
# computing
pip install --upgrade cython "numpy>=1.21.0"
pip install "scikit-fmm==2022.03.26" scikit-learn scikit-image
pip install einops gym omegaconf
pip install matplotlib seaborn pandas setproctitle
# pytorch related
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install "tensorboard~=2.8.0"
pip install pytorch-lightning
pip install segmentation_models_pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install "dungeon_maps==0.0.3a1"
pip install "git+https://github.com/Ending2015a/rlchemy"
# Resolve distutils has no attribute version issue
pip install "setuptools==59.5.0"
