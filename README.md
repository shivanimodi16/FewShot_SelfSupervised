# FewShot_SelfSupervised

In this research, we examine the few-shot capabilities of two well-known self-supervised learning algorithms for visual representations, SimCLR and SimSiam, and see how they stack up against their supervised counterpart. A common evaluation protocol is to train a linear classifier on top of (frozen) representations learnt by self-supervised methods. We take this protocol a step further by evaluating our supervised and self-supervised methods on a few-shot image classification task using frozen representations. In our Experiments, we find that as expected the supervised method has a higher performance than self-supervised methods SimCLR and SimSiam.

<img src="https://user-images.githubusercontent.com/22906652/146695744-2ce62b38-55b2-444b-b878-d1fcfcef5b84.png" width="500" height="200">

The  performance of few-shot transfer via linear evaluation for image classification on CIFAR10 and STL10. 

<img width="688" alt="Screen Shot 2021-12-19 at 4 35 57 PM" src="https://user-images.githubusercontent.com/22906652/146699754-eea55a58-425a-4cae-ae26-2dfa7586bc7a.png">


