# Droid Epipolar

Adding Epipolar Constraints to DROID-SLAM training for improved geometric consistency.

## Setup

### Clone repo

```bash
git clone --recursive https://github.com/eggonz/droid-epipolar.git
cd droid-epipolar
```

#### Submodules

If you don't specify `--recursive` git submodules will not get installed.
In that case, you can run:
```bash
git submodule init
git submodule update
```

### Install env

Create `droidenv` conda environment:

```bash
cd scripts
bash -i scripts/conda_create_droidenv.sh

source activate droidenv

cd ../3rdparty/DROID_SLAM
python -m pip install evo --upgrade --no-binary evo
python -m pip install gdown
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 -f https://download.pytorch.org/whl/torch_stable.html
CUDA_HOME=/usr/local/cuda-10.1 python setup.py install
```

### Download dataset

TartanAir:

1. Make sure you have the [tartanair_tools](https://github.com/castacks/tartanair_tools) repository placed under `3rdparty/`
2. Run `scripts/download_tartanair.sh`

## Training

### Example script

See `scripts/train.sh`

### Scripts with experiment configurations

See `scripts/experiments/exp*.job`

### Notes on DROID-SLAM training

- database cache: a pickle file for the dataset is created at `droid-epipolar/src/droid_slam/dara_readers/cache/TartanAir.pickle`
- saved model checkpoints: model checkpoints are generated with the experiment name and saved at `droid-epipolar/src/checkpoints/EXPNAME_STEPNUMBER.pth`

## Checkpoints

You can find example data and checkpoints under `data/`, including the initial DroidNet checkpoint, the trained checkpoints for each experiment and the dataset pickle files for TartanAir example subset.
