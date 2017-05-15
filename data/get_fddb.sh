#!/usr/bin/env sh
# This script downloads the FDDB data and unzips it

mkdir FDDB
cd FDDB
mkdir images
cd images
echo "Downloading images..."
curl "tamaraberg.com/faceDataset/originalPics.tar.gz" -o images.tar.gz
echo "Unziping images..."
tar -xvzf images.tar.gz > /dev/null 2>&1
rm images.tar.gz
cd ..
echo "Downloading annotations..."
curl "vis-www.cs.umass.edu/fddb/FDDB-folds.tgz" -o annotations.tgz
echo "Unziping annotations..."
tar -xvzf annotations.tgz > /dev/null 2>&1
rm annotations.tgz
cd ..



