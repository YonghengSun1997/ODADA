## Rethinking adversarial domain adaptation: Orthogonal decomposition for unsupervised domain adaptation in medical image segmentation
This repository provides the code for "Rethinking adversarial domain adaptation: Orthogonal decomposition for unsupervised domain adaptation in medical image segmentation". Our work is accepted by MedIA [mia_link].



[mia_link]:https://www.sciencedirect.com/science/article/abs/pii/S1361841522002511

![ODADA](./pictures/model.png)
Fig. 1. Structure of ODADA.



### Requirementss
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* Python == 3.7 
* Some basic python packages such as Numpy.

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

## Dataset
This Repository contains a toy dataset (HK and BIDMC) for reimplement. You can download a full-version dataset via
https://drive.google.com/drive/folders/1KEomtcpTUYCc94nAvEBBsT3vvLnR4rPN?usp=share_link
If the data violates privacy, please let us know in time.
## Usages
### For multi-site prastate segmentation



1. To train ODADA for multi-site prostate segmentation, run:
```
python main.py
```

2. Our experimental results are shown in the table:
![refinement](./pictures/results.png)


## Citation
If you find our work is helpful for your research, please consider to cite:
```
@article{sun2022rethinking,
  title={Rethinking adversarial domain adaptation: Orthogonal decomposition for unsupervised domain adaptation in medical image segmentation},
  author={Sun, Yongheng and Dai, Duwei and Xu, Songhua},
  journal={Medical Image Analysis},
  pages={102623},
  year={2022},
  publisher={Elsevier}
}
```
