"""
A robust bbox regression loss function.
"""

import argparse
import caffe
import numpy as np

class RobustL1LossLayer(caffe.Layer):

    @classmethod
    def parse_args(cls, argStr):
        parser = argparse.ArgumentParser(description='FullImInputLayer')
        parser.add_argument('--width', default=0, type=float)
        args = parser.parse_args(argStr.split())
        return args

    def setup(self, bottom, top):
        param = RobustL1LossLayer.parse_args(self.param_str)
        self.width = param.width

    def reshape(self, bottom, top):
        assert bottom[0].data.shape == bottom[1].data.shape
        top[0].reshape()

    def forward(self, bottom, top):
        d = bottom[0].data[...] - bottom[1].data[...]
        w = self.width
        loss = np.maximum(np.abs(d) - w, 0)
        if len(d.shape) == 2: # RCNN
            n = d.shape[0]
            mult = np.where(bottom[1].data != 0, 1, 0)
            top[0].data[...] = np.sum(loss * mult) / n
        else: # RPN
            n = d.shape[1] * d.shape[2] * d.shape[3] / 4
            top[0].data[...] = np.sum(loss) / n

    def backward(self, top, propagate_down, bottom):
        d = bottom[0].data[...] - bottom[1].data[...]
        w = self.width
        grad = np.where(np.abs(d) > w, np.sign(d), 0)
        if len(d.shape) == 2: # RCNN
            n = d.shape[0]
            mult = np.where(bottom[1].data != 0, 1, 0)
            bottom[0].diff[...] = grad * mult / n
        else: # RPN
            n = d.shape[1] * d.shape[2] * d.shape[3] / 4
            bottom[0].diff[...] = grad / n
