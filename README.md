# BotSpot:A Hybrid Learning Framework to Uncover Bot Install Fraud in Mobile Advertising
This is the implementation of BotSpot - A Hybrid Learning Framework to Uncover Bot Install Fraud in Mobile Advertising.
For more details, please refer to our paper. https://dl.acm.org/doi/abs/10.1145/3340531.3412690


To replicate:
1. download dataset from: https://drive.google.com/file/d/1KPJnfj4A7UdRds0hPWPSRKU_CKGwoMyI/view?usp=sharing, and de-compress the zip file to the root diretory. we have dataset-1 and dataset-2 with sub-folders inside
2. run python XXX.py [dataset]， e.g., python baseline_mlp.py dataset-1 to train MLP model for dataset 1, similarly for all other baseline methods and botspot.py
3. Watch out, to switch to the second dataset,first uncomment the hard-coded id values for the second dataset in model_main.py and model_main_GAT_baseline.py


## Cite
Please cite our paper as below if you use this code in your work:

```
@article{Yao2020BotSpotAH,
  title={BotSpot: A Hybrid Learning Framework to Uncover Bot Install Fraud in Mobile Advertising},
  author={Tianjun Yao and Q. Li and Shangsong Liang and Yadong Zhu},
  journal={Proceedings of the 29th ACM International Conference on Information & Knowledge Management},
  year={2020},
  url = {https://dl.acm.org/doi/abs/10.1145/3340531.3412690},
  doi = {10.1145/3340531.3412690}
}
```
