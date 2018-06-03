"""
A robust bbox regression loss function.
"""

import argparse
import caffe
import numpy as np

class RobustL1LossLayer(caffe.Layer):

    @classmethod
    def parse_args(cls, argStr):
        parser = argparse.ArgumentParser(description='RobustL1LossLayer')
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

class RobustL1LossLayer2(caffe.Layer):

    @classmethod
    def parse_args(cls, argStr):
        parser = argparse.ArgumentParser(description='RobustL1LossLayer2')
        parser.add_argument('--widthxy', default=0, type=float)
        parser.add_argument('--widthwh', default=0, type=float)
        args = parser.parse_args(argStr.split())
        return args

    def setup(self, bottom, top):
        param = RobustL1LossLayer2.parse_args(self.param_str)
        self.widthxy = param.widthxy
        self.widthwh = param.widthwh

    def reshape(self, bottom, top):
        assert bottom[0].data.shape == bottom[1].data.shape
        top[0].reshape()

    def forward(self, bottom, top):
        w = np.array([self.widthxy, self.widthxy, self.widthwh, self.widthwh])
        s = bottom[0].data.shape
        if len(s) == 2: # RCNN
            d = bottom[0].data[...].reshape([-1, 4]) - \
                bottom[1].data[...].reshape([-1, 4])
            loss = np.maximum(np.abs(d) - w, 0).reshape(s)
            n = s[0]
            mult = np.where(bottom[1].data != 0, 1, 0)
            top[0].data[...] = np.sum(loss * mult) / n
        else: # RPN
            d = bottom[0].data[...].reshape(s[0], -1, 4, s[2], s[3]) - \
                bottom[1].data[...].reshape(s[0], -1, 4, s[2], s[3])
            w = w[:, np.newaxis, np.newaxis]
            loss = np.maximum(np.abs(d) - w, 0).reshape(s)
            n = s[1] * s[2] * s[3] / 4
            top[0].data[...] = np.sum(loss) / n

    def backward(self, top, propagate_down, bottom):
        w = np.array([self.widthxy, self.widthxy, self.widthwh, self.widthwh])
        s = bottom[0].data.shape
        if len(s) == 2: # RCNN
            d = bottom[0].data[...].reshape([-1, 4]) - \
                bottom[1].data[...].reshape([-1, 4])
            grad = np.where(np.abs(d) > w, np.sign(d), 0).reshape(s)
            n = s[0]
            mult = np.where(bottom[1].data != 0, 1, 0)
            bottom[0].diff[...] = grad * mult / n
        else: # RPN
            d = bottom[0].data[...].reshape(s[0], s[1]/4, 4, s[2], s[3]) - \
                bottom[1].data[...].reshape(s[0], s[1]/4, 4, s[2], s[3])
            w = w[:, np.newaxis, np.newaxis]
            grad = np.where(np.abs(d) > w, np.sign(d), 0).reshape(s)
            n = s[1] * s[2] * s[3] / 4
            bottom[0].diff[...] = grad / n
