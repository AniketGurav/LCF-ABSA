# LCF-ABSA
The LCF-BERT model also can be found at [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch)

> Pytorch Implementations.

> Pytorch-transformers.

> Aspect-based Sentiment Analysis (ABSA/ABSC).


## Requirement
* python 3.7
* pytorch >=1.0
* [Pytorch-transformers](https://github.com/huggingface/transformers)
* To unleash peroformance of LCF-BERT models, a GTX 1080Ti or other GPU equipped with large memory is required.

## Datasets

* SemEval-2014 (Resetaurant and Laptop datasets) 
* ACL twitter dataset

## Train

Train the model by

```
python train.py --model lcf_bert --dataset laptop --SRD 3 --local_context_focus cdm --use_single_bert
```

or try to train in batchs

```
python batch_train.py --config experiments.json
```

 Try to assign *use_single_bert = true* while out-of-memory error occurs.

## Performance of LCF design models
Performance based on the pytorch pretrained model of bert-base-uncased.

| Models           | Restaurant (acc) | Laptop (acc) |  Twitter(acc) 
| ------------- | :-----:| :-----:| --- | 
| LCF-Glove-CDM | 82.50 | 76.02 | 72.25| 
| LCF-Glove-CDW | 81.61 | 75.24 | 71.82| 
| LCF-BERT-CDM | 86.52 | 82.29 | 76.45| 
| LCF-BERT-CDW | 87.14 | 82.45 | 77.31| 

Generally, the best performance needs several independent training processes.

### For Better Performance
This repository can achieving superior performance with [BERT-ADA](https://arxiv.org/pdf/1908.11860.pdf) pretrained models. Learn to train the domain adapted BERT pretraiend models from [domain-adapted-atsc](https://github.com/deepopinion/domain-adapted-atsc), and place the pretrained models in *bert_pretrained_models*. The results in following table are the best of five trainings (random seed 0, 1, 2, 3, 4). Try set other random seeds to explore different results.

| Models            | Restaurant (acc)  | Laptop (acc)  |  Twitter(acc) 
| -------------     | :-----:           | :-----:       | ---           | 
| LCF-BERT-CDM      |                   | 82.92         |               | 
| LCF-BERT-CDW      |                   |               |               | 
| LCF-BERT-Fusion   |                   |               | 77.17         | 

The state-of-the-art benchmarks of the ABSA task can be found at [NLP-progress](https://nlpprogress.com) (See Section of SemEval-2014 subtask4)

## Acknowlegement

Our work is based on the repositories of [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and the [Pytorch-transformers](https://github.com/huggingface/transformers). Thanks to the authors for their devotion and Thanks to everyone who offered assistance.
Feel free to report any bug or discuss with us. 

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

