# CIKM-Applied_Research-2150
This repository contains code and dataset links for CIKM-2150: BotSpot:A Hybrid Learning Framework to Uncover Bot Install Fraud in Mobile Advertising
To replicate:
1. download dataset from: https://drive.google.com/file/d/1KPJnfj4A7UdRds0hPWPSRKU_CKGwoMyI/view?usp=sharing, and de-compress the zip file to the root diretory. we have dataset-1 and dataset-2 with sub-folders inside
2. run python XXX.py [dataset]， e.g., python baseline_mlp.py dataset-1 to train MLP model for dataset 1, similarly for all other baseline methods and botspot.py
3. Be careful, to switch to the second dataset,first uncomment the hard-coded id values in model_main.py and model_main_GAT_baseline.py
