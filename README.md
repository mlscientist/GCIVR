# GCIVR
This repository implements GCIVR algorithm on 3 tasks. This code is for ICLR submission and review, hence, any reuse outside this review process is prohibited.

## Getting Started
To run this code, you need to have PyTorch installed. You can install the requirements for this repo using the `requirements.txt` with:
```cli
pip install -r requirements.txt
```


#### 1- DRO Fairness
To run a sample code please run the following command, which runs the constrained optimization using GCIVR. You may change parameters of `1-DRO_fairness.py` file to tune the training as well. The `Adult` dataset will be downloaded automatically.
```cli
 python 1-DRO_fairness.py -c
 ```


#### 2- Intersectional Fairness
To run a sample code please run the following command, which runs the constrained optimization using GCIVR. You may change parameters of `2-intersectional_fairness.py` file to tune the training as well. The `Communities and Crime` dataset will be downloaded automatically.
```cli
 python 2-intersectional_fairness.py -c
 ```
 
 
 #### 3- Ranking Fairness
To run a sample code please run the following command, which runs the constrained optimization using GCIVR. You may change parameters of `3-ranking_fairness.py` file to tune the training as well. You need to download the Microsoft `Learning to Rank` Datset from [MSLR-WEB10K](https://1drv.ms/u/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78) and point to the directory where the files `train.txt` and `test.txt` resided in the syntax with flag `-dp`.
```cli
 python 3-ranking_fairness.py -c -dp [Path_to_data_folder]
 ```
