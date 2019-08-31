# LCF-ABSA

> Pytorch Implementations.

> Aspect-based Sentiment Analysis (ABSA/ABSC)


## Requirement
* python 3.7
* pytorch >=1.0
* [pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/) >= 0.6.1
* For LCF-BERT models, a GTX 1080Ti or other GPU equipped with large memory is required.

## Datasets

* SemEval-2014 (Resetaurant and Laptop datasets) 
* ACL twitter dataset

## Train


```
python train.py --model lcf_bert --dataset laptop --SRD 3 --local_context_focus cdm
```


## Performance of LCF design models

| Models           | Restaurant (acc) | Laptop (acc) |  Twitter(acc) 
| ------------- | :-----:| :-----:| --- | 
| LCF-Glove-CDM | 82.50 | 76.02 | 72.25
| LCF-Glove-CDW | 81.61 | 75.24 | 71.82
| LCF-BERT-CDM | 86.52 | 82.29 | 76.45
| LCF-BERT-CDW | 87.14 | 82.45 | 77.31

Generally, the best performance needs several independent training processes.

The state-of-the-art benchmarks of the ABSA task can be found at [NLP-progress](https://nlpprogress.com) (See Section of SemEval-2014 subtask4)

## Notice

This repository is the raw code for [LCF: A Local Context Focused Aspect-based Sentiment Classiﬁcation with Pre-trained BERT](https://www.mdpi.com/2076-3417/9/16/3389/pdf), and unexpected problem may occurs on different platforms.

## Acknowlegement

Our work is based on the repositories [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and the pytorch-pretrained-bert. Thanks to the authors for their devotion and Thanks to all the scholars who offered assistance.
