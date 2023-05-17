# Domain Generalization - Confident Experts

This repository is the implementation of paper **Domain Generalization - Confident Experts**



### Environment Setup
Python version : `3.8.10`  
GPU : *NVIDIA V100 GPU*  
Install the required libraries using 
```sh
pip3 install -r requirements.txt
```

### Downloading Datasets
The paper uses the following datasets:
1. PACS
2. VLCS
3. TerraIncognita
4. Office-Home 
5. SVIRO

The datasets can be downloaded using the following command. (Uncomment the lines corresponding to required datasets in `domainbed/scripts/download`)
```sh
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data
```

## Training

Train single model:

```sh
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/SVIRO/\
       --algorithm COnfidentExperts\
       --dataset SVIRO\
       --test_env 2
```
Train sweep of models:
```sh
python -m domainbed.scripts.sweep launch\
	        --data_dir=../../MoE-DG/domainbed/data/ \
		        --command_launcher multi_gpu\
			     	--output_dir train_output\
			            --algorithms ConfidentExperts\
				           --datasets SVIRO\
							--n_hparams 3\
						     	--n_trials 1\
							 		--skip_confirmation\
										--single_test_envs
```
To train a model using routings(*clustered* or *stratified*) export appropriate environment variables before running training scripts. 
```sh
export USE_ROUTING=1 # required for using any routing mechanism
export USE_STRATIFIED=1 # use this to replace clustered routing with stratified routing
export USE_RANDOM_ROUTING=1 # use this to replace clustered routing with random routing
```
### Clustering 
To obtain the *cluster* and *stratified* mapping for each dataset, use the below command:
```sh
python -m domainbed.scripts.clustering --data_dir ../../MoE-DG/domainbed/data/ --dataset OfficeHome --batch_size 16 --test_env 0
```
(change the dataset and test_env based on experiments)  
Soon these mapping/routing tensors will also be released here.

### Evaluation
To evaluate a trained model, set the `PATH` variable in python script to the desired model and run the following command
```sh
python3 -m domainbed.scripts.evaluate --data_dir=../../MoE-DG/domainbed/data/ --algorithm ConfidentExperts --dataset PACS --test_env 0
```

### Results
Deterministic routing
| Algorithm      | PACS | VLCS  | OfficeHome  | TerraIncognita | 
| :---:        |    :----:   |          :---: | :---: | :---: |
| MIRO      | 79.0  | 85.4 | 70.5  |  **50.4**  |
| GMOE      |   **80.2**   |  88.1  |  74.2  |  48.5  |
| CE (ours, clus.)      | 79.9       | 87.1   |  **74.4** |  40.8  |
| CE (ours, stra.)      | 80.0 | **88.4**  | 73.8  |  42.8  |

Random routing on large datasets
| Algorithm      | SVIRO | 
| :---:        |    :----:  |
| ViT      |   89.6    | 
| GMOE     |    90.3   | 
|  CE (ours)     |   **92.2**   | 

### Hyperparameters
Default hyperparameters of the models of respective datasets are set to optimal ones. (based on train-domain validation)

### Pre-trained models
To be released soon

### Acknowledgements
1. GMOE : https://github.com/Luodian/Generalizable-Mixture-of-Experts
2. DomainBed : https://github.com/facebookresearch/DomainBed

