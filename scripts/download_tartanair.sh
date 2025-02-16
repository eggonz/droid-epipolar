#!/bin/bash

DATA_ROOT=""

# 1. Download

#pip install boto3 colorama minio

OUTDIR=$DATA_ROOT/epipolar_data/datasets/TartanAir_download
mkdir -p $OUTDIR

cd $HOME/droid-epipolar/3rdparty/tartanair_tools
python download_training.py --output-dir=$OUTDIR --rgb --depth --only-left --cloudflare

# 2. Unzip

SRC=$DATA_ROOT/epipolar_data/datasets/TartanAir_download
DST=$DATA_ROOT/epipolar_data/datasets/TartanAir
mkdir -p $DST

FILES=$(find $SRC -maxdepth 1 -type f -name "*.zip")

for f in $FILES
do
    echo "Unzipping $f"
    unzip -o $f -d $DST
    echo "Deleting $f"
    rm $f
done

# find $DST -type f -name "*right*" -delete

# 3. Example subset

SUBSET="
abandonedfactory/Easy/P000
"
for s in $SUBSET
do
    src=$DATA_ROOT/epipolar_data/datasets/TartanAir/$s
    dst=$DATA_ROOT/epipolar_data/datasets/TartanAir_example/$s

    echo "Copying $src to $dst"
    mkdir -p $dst
    cp -nvr $src/* $dst
done
