# run using: `bash -i conda_create_droidenv.sh`

conda create --yes -n droidenv python=3.9
conda activate droidenv

conda install --yes -c open3d-admin open3d=0.15.1
conda install --yes -c rusty1s -c pytorch -c nvidia -c conda-forge -c defaults cudatoolkit=11.3.1
conda install --yes -c rusty1s -c pytorch -c nvidia -c conda-forge -c defaults tensorboard=2.15.1
conda install --yes -c rusty1s -c pytorch -c nvidia -c conda-forge -c defaults scipy=1.11.3
conda install --yes -c rusty1s -c pytorch -c nvidia -c conda-forge -c defaults tqdm=4.66.1
conda install --yes -c rusty1s -c pytorch -c nvidia -c conda-forge -c defaults suitesparse=5.10.1
conda install --yes -c rusty1s -c pytorch -c nvidia -c conda-forge -c defaults matplotlib=3.8.0
conda install --yes -c rusty1s -c pytorch -c nvidia -c conda-forge -c defaults pyyaml=6.0.1

# VIA MODULE
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2  # cu117
pip install opencv-python
pip install wandb