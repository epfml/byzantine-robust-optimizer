# Learning from History for Byzantine Robust Optimization

This repository contains research code for our [Byzantine robust optimization paper](https://arxiv.org/abs/2012.10333).


Summary:
We study the problem of federated and distributed learning in the presence of untrusted workers who may try to derail the training process. We first describe a simple new aggregator based on iterative **centered clipping** which has significantly stronger theoretical guarantees than previous methods. This aggregator is especially interesting since, unlike most preceding methods, it is very scalable requiring only *O(n)* computation and communication per round. Further, it is also compatible with other strategies such as [asynchronous updates](https://arxiv.org/abs/1604.00981) and [secure aggregation](https://eprint.iacr.org/2017/281.pdf), both of which are crucial for real world applications. Secondly, we show that the time coupled attacks can easily be overcome by using **worker momentum**. 


# Code organization

### A few pointers


### Distributed training & changing config


# Reference

If you use this code, please cite the following [paper](https://arxiv.org/abs/2012.10333)

    @inproceedings{karimireddy2020learning,
      author = {Karimireddy, Sai Praneeth and He, Lie and Jaggi, Martin},
      title = "{Learning from History for Byzantine Robust Optimization}",
      booktitle = {ICML 2021 - Proceedings of International Conference on Machine Learning},
      year = 2021,
      url = {https://arxiv.org/abs/2012.10333}
    }

