# tnarihi's Chainer extensions

I just started playing with [Chainer](http://chainer.org/) which is a deep learning package in Python (core numerical computation is written in C++/CUDA) developed by Preferred Networks. It seems to be convenient for fast prototyping of neural networks for research, especially for recurrent things, due to "define-by-run" strategy and dynamically allocating memory. Maybe I am going to put my stuff for extending Chainer to this repo.

## Functions
### Deconvolution2D
This is defined as an inverted operation of Convolution which is effectively used in [Fully Convnets](http://arxiv.org/abs/1411.4464) for (learned) upsampling strided output maps. I referred to [an implementation in Caffe](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/deconv_layer.cpp) to implement this.
