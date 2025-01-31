# CosFace_pytorch


## Goal of this repo: **Transfer this model to a updated environment**

***Pytorch implementation of CosFace***

Old Requirement

------------

- Deep Learning Platform:  PyTorch 0.4.1
- OS:  CentOS Linux release 7.5
- Language:  Python 2.7
- CUDA:  8.0

------------

-  Database:  `WebFace` or `VggFace2` (You should first complete the data preprocessing section by following these steps https://github.com/wy1iu/sphereface#part-1-preprocessing)
- Network:  `sphere20`, `sphere64`, `LResnet50E-IR`(In ArcFace paper)

------------

## Previous Result

### Result(new)

Single model trained on CAISA-WebFace achieves **~99.2%** accuracy on LFW ( [Downlaod Link](https://pan.baidu.com/s/1uOBATynzBTzZwrIKC4kcAA) Password: 69e6)

**NOTE: PyTorch has already been upgraded to 1.1x, and I would try to transfer the whole repo into a up-to-date environment**


Commented at 2018 by the original author
------------
Note: Pytorch 0.4 seems to be very different from 0.3, which leads me to not fully reproduce the previous results. Currently still adjusting parameters....
The initialization of the fully connected layer does not use Xavier but is more conducive to model convergence.


### Result(old)

Network  |  Hyper-parameter  |  Accuracy on LFW
------------- | -------------  |  -------------
Sphere20  | s=30, m=0.35  |  99.08%
Sphere20  | s=30, m=0.40  |  99.23%
LResnet50E-IR(In ArcFace paper)  | s=30, m=0.35  |  99.45%
