# LCF-ABSA

Here is new paper aims at Chinese and multilingual-oriented Aspect Polarity Classification and Aspect Term Extraction:
[A Multi-task Learning Model for Chinese-oriented Aspect Polarity Classification and Aspect Term Extraction](https://arxiv.org/abs/1912.07976), the codes are available at: [LCF-ATEPC](https://github.com/yangheng95/LCF-ATEPC).

code for paper [LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification](https://www.mdpi.com/2076-3417/9/16/3389).

This is the up-to-date version which transferred pytorch-pretrained-bert to pytorch-transformers, and an earlier version of LCF-BERT models can be found at [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).

> Pytorch Implementations.

> Pytorch-transformers.

> Aspect-based Sentiment Analysis (ABSA/ABSC).


## Requirement
* python 3.7
* pytorch >=1.0
* [Pytorch-transformers](https://github.com/huggingface/transformers)
* To unleash the performance of LCF-BERT models, a GTX 1080Ti or other GPU equipped with a large memory is required. 
* `In fact, the memory consumption of this model can be significantly optimized in several ways (minize to 6GB of RAM).`
## Datasets

* SemEval-2014 (Restaurant and Laptop datasets) 
* ACL twitter dataset

## Train

Train the model with cmd

```
python train.py --model lcf_bert --dataset laptop --SRD 3 --local_context_focus cdm --use_single_bert
```

or try to train in batches

```
python batch_training.py --config experiments.json
```

 Try to set '*batch_size=8*', *'max_seq_len = 40'* or *'use_single_bert = True'* while out-of-memory error occurs.

## Performance of LCF design models
The performance based on the pytorch pre-trained model of bert-base-uncased.

| Models           | Restaurant (acc) | Laptop (acc) |  Twitter(acc) 
| ------------- | :-----:| :-----:| --- | 
| LCF-Glove-CDM | 82.50 | 76.02 | 72.25| 
| LCF-Glove-CDW | 81.61 | 75.24 | 71.82| 
| LCF-BERT-CDM | 86.52 | 82.29 | 76.45| 
| LCF-BERT-CDW | 87.14 | 82.45 | 77.31| 

### For Better Performance
This repository can achieve superior performance with [BERT-ADA](https://arxiv.org/pdf/1908.11860.pdf) pre-trained models. Learn to train the domain adapted BERT pretrained models from [domain-adapted-atsc](https://github.com/deepopinion/domain-adapted-atsc), and place the pre-trained models in bert_pretrained_models. The results in the following table are the best of five training processes (random seed 0, 1, 2, 3, 4), Refer to the training [log](./train.log) to reproduce the results. Try to set other random seeds to explore different results.

| Models            | Restaurant (acc)  | Laptop (acc)  |  Twitter(acc) 
| -------------     | :-----:           | :-----:       | ---           | 
| LCF-BERT-CDM      | 89.11             | 82.92         | 77.89         | 
| LCF-BERT-CDW      | 89.38             | 82.76         | 77.17         | 
| LCF-BERT-Fusion   | 88.95             | 82.45         | 77.75         | 

The state-of-the-art benchmarks of the ABSA task can be found at [NLP-progress](https://nlpprogress.com) (See Section of SemEval-2014 subtask4)

## Acknowlegement

Our work is based on the repositories of [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and the [Pytorch-transformers](https://github.com/huggingface/transformers). Thanks to the authors for their devotion and Thanks to everyone who offered assistance.
Feel free to report any bug or discussing with us. 

## Citation
If this repository is helpful to you, please cite our paper:

    @article{zeng2019lcf,
        title={LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification},
        author={Zeng, Biqing and Yang, Heng and Xu, Ruyang and Zhou, Wu and Han, Xuli},
        journal={Applied Sciences},
        volume={9},
        number={16},
        pages={3389},
        year={2019},
        publisher={Multidisciplinary Digital Publishing Institute}
    }

## Reference 

[Adapt or Get Left Behind: Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification](https://arxiv.org/pdf/1908.11860.pdf)

