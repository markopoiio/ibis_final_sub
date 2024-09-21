# IBIS challenge submission, team mj

## Prepare env

conda create -n ibis_mj

conda activate ibis_mj

conda install tensorflow=2.15

pip install keras-tcn

## Gather datasets:
* unzip train data to "train" fold
* put hg38 genome to "hg38.fa" file
* run dataset-datherer

## Model training:
* run g2a_trainer_final.py and a2g_trainer_final.py

## Predicting:
* unzip test data to "test_final" fold
* run predictions.ipynb
