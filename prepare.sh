#!/bin/bash

pip install -r requirements.txt

mkdir -p ./.data/multi30k/raw
wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/train.en.gz && mv train.en.gz ./.data/multi30k/raw && gzip -d ./.data/multi30k/raw/train.en.gz
wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/train.de.gz && mv train.de.gz ./.data/multi30k/raw && gzip -d ./.data/multi30k/raw/train.de.gz
wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/val.en.gz && mv val.en.gz ./.data/multi30k/raw && gzip -d ./.data/multi30k/raw/val.en.gz
wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/val.de.gz && mv val.de.gz ./.data/multi30k/raw && gzip -d ./.data/multi30k/raw/val.de.gz
wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/test_2016_flickr.en.gz && mv test_2016_flickr.en.gz ./.data/multi30k/raw && gzip -d ./.data/multi30k/raw/test_2016_flickr.en.gz
wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/test_2016_flickr.de.gz && mv test_2016_flickr.de.gz ./.data/multi30k/raw && gzip -d ./.data/multi30k/raw/test_2016_flickr.de.gz
