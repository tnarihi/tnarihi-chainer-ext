#! /usr/bin/env python
"""
"""
from __future__ import division
from __future__ import print_function

import argparse

from scipy.io import savemat

from chainer.functions.caffe import CaffeFunction

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='Path to caffemodel of VGG-xx layers')
parser.add_argument('mat',
	help='Path to output matlab file. Parent folder should exists beforehand.')

args = parser.parse_args()

# Main
print('Loading VGG caffemodel: {}'.format(args.caffemodel))
caffevgg = CaffeFunction(args.caffemodel)
print('Done.')
print('Converting')
fs = caffevgg.fs
params = {}
for i in range(8):
    j = 0
    while True:
        try:
            key = 'conv{}_{}'.format(i + 1, j + 1)
            param = getattr(fs, key)
            params[key + '_W'] = param.W.copy()
            params[key + '_b'] = param.b.copy()
        except:
            break
        j += 1
    if j == 0:
        key = 'fc{}'.format(i + 1)
        param = getattr(fs, key)
        params[key + '_W'] = param.W.copy()
        params[key + '_b'] = param.b.copy()
print('Done.')
print('Saving to: {}'.format(args.mat))
savemat(args.mat, params)
print('Done.')
