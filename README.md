# tnarihi's Chainer extensions

I just started playing with [Chainer](http://chainer.org/) which is a deep learning package in Python (core numerical computation is written in C++/CUDA) developed by Preferred Networks. It seems to be useful for fast prototyping of neural networks research, especially for recurrent things, due to "define-by-run" strategy and dynamically allocating memory. Maybe I am going to put my stuff for extending Chainer to this repo.

## Dependencies

* Chainer >= 1.1.1

## Functions
### Deconvolution2D
This is defined as an inverted operation of Convolution which is effectively used in [Fully Convnets](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) for (learned) upsampling strided output maps. I referred to [an implementation in Caffe](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/deconv_layer.cpp) to implement this.

### Maximum/Minimum
Maximum/Minimum takes two arrays with the same shape and behaves like `numpy.maximum` and `cuda.gpuarray.maximum`. Since these are not parameterized, you can use the shortcut functions `maximum` and `minimum` in the `layers` module.


## Utilities

Utility functions/classes are in the `utils` module.

### data_provider
A generator function which yields a tuple of mini-batches infinitely. Each generator call create threads for prefetching data in the background. See doc.

### blob_to_tile
This transforms an array with shape of `(b, c, h, w)` into a gray image consists of tiled `b * n` images except for c=3 case, which outputs a color image with tiled `b` images. This is useful for visualizing inputs, outputs, feature maps and filters in vision tasks.

## Testing

Go to `tests` folder, then run:

```
PYTHONPATH=.. nodetests
```
