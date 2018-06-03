"""
Unit tests for the robust bbox regression loss function.
"""

import os

os.environ['GLOG_minloglevel'] = '2'

import caffe
import numpy as np
import tempfile
import unittest

EPSILON = 1e-4

def robust_l1_loss_layer_net_file():
    netStr = """
        name: "test"
        force_backward: true
        input: "pred"
        input_shape { dim: 3 dim: 3 }
        input: "label"
        input_shape { dim: 3 dim: 3 }
        layer {
            name: "loss"
            type: "Python"
            bottom: "pred"
            bottom: "label"
            top: "loss"
            python_param {
                module: "robust_loss_layer.layer"
                layer: "RobustL1LossLayer"
                param_str: "--width=1"
            }
        }"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(netStr)
    return f.name

class RobustL1LossLayerTestCase(unittest.TestCase):

    def setUp(self):
        net_file = robust_l1_loss_layer_net_file()
        self.net = caffe.Net(net_file, caffe.TEST)
        os.remove(net_file)

    def test_far(self):
        pred = np.ones([3, 3])
        label = 5. * np.ones([3, 3])
        self.net.blobs['pred'].data[...] = pred
        self.net.blobs['label'].data[...] = label
        self.net.forward()
        self.net.backward()
        loss = self.net.blobs['loss'].data[...]
        actual_loss = 9.
        diff = self.net.blobs['pred'].diff[...]
        actual_diff = -1./3 * np.ones([3, 3])
        self.assertTrue(abs(loss - actual_loss) < EPSILON)
        self.assertTrue((abs(diff - actual_diff) < EPSILON).all())
        pred = np.ones([3, 3])
        label = 1.5 * np.ones([3, 3])
        self.net.blobs['pred'].data[...] = pred
        self.net.blobs['label'].data[...] = label
        self.net.forward()
        self.net.backward()
        loss = self.net.blobs['loss'].data[...]
        actual_loss = 0.
        diff = self.net.blobs['pred'].diff[...]
        actual_diff = np.zeros([3, 3])
        self.assertTrue(abs(loss - actual_loss) < EPSILON)
        self.assertTrue((abs(diff - actual_diff) < EPSILON).all())

    def test_mixed(self):
        pred = np.ones([3, 3])
        label = np.outer([-2, 0, 2], np.ones(3))
        self.net.blobs['pred'].data[...] = pred
        self.net.blobs['label'].data[...] = label
        self.net.forward()
        self.net.backward()
        loss = self.net.blobs['loss'].data[...]
        actual_loss = 2.
        diff = self.net.blobs['pred'].diff[...]
        actual_diff = np.outer([1./3, 0, 0], np.ones(3))
        self.assertTrue(abs(loss - actual_loss) < EPSILON)
        self.assertTrue((abs(diff - actual_diff) < EPSILON).all())

    def test_label_diff(self):
        pred = np.ones([3, 3])
        label = np.outer([-2, 0, 2], np.ones(3))
        self.net.blobs['pred'].data[...] = pred
        self.net.blobs['label'].data[...] = label
        self.net.forward()
        self.net.backward()
        label_diff = self.net.blobs['label'].diff[...]
        self.assertFalse(np.any(label_diff))


def robust_l1_loss_layer_2_rcnn_net_file():
    netStr = """
        name: "test"
        force_backward: true
        input: "pred"
        input_shape { dim: 1 dim: 8 }
        input: "label"
        input_shape { dim: 1 dim: 8 }
        layer {
            name: "loss"
            type: "Python"
            bottom: "pred"
            bottom: "label"
            top: "loss"
            python_param {
                module: "robust_loss_layer.layer"
                layer: "RobustL1LossLayer2"
                param_str: "--widthxy=1 --widthwh=2"
            }
        }"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(netStr)
    return f.name

def robust_l1_loss_layer_2_rpn_net_file():
    netStr = """
        name: "test"
        force_backward: true
        input: "pred"
        input_shape { dim: 3 dim: 8 dim: 1 dim: 1}
        input: "label"
        input_shape { dim: 3 dim: 8 dim: 1 dim: 1}
        layer {
            name: "loss"
            type: "Python"
            bottom: "pred"
            bottom: "label"
            top: "loss"
            python_param {
                module: "robust_loss_layer.layer"
                layer: "RobustL1LossLayer2"
                param_str: "--widthxy=1 --widthwh=2"
            }
        }"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(netStr)
    return f.name

class RobustL1LossLayer2TestCase(unittest.TestCase):

    def test_rcnn(self):
        net_file = robust_l1_loss_layer_2_rcnn_net_file()
        net = caffe.Net(net_file, caffe.TEST)
        os.remove(net_file)
        pred = np.ones([1, 8])
        label = np.zeros([1, 8])
        label[:, :4] = 5.
        net.blobs['pred'].data[...] = pred
        net.blobs['label'].data[...] = label
        net.forward()
        net.backward()
        loss = net.blobs['loss'].data[...]
        actual_loss = 10.
        diff = net.blobs['pred'].diff[...]
        actual_diff = np.zeros([1, 8])
        actual_diff[:, :4] = -1.
        self.assertTrue(abs(loss - actual_loss) < EPSILON)
        self.assertTrue((abs(diff - actual_diff) < EPSILON).all())

    def test_rpn(self):
        net_file = robust_l1_loss_layer_2_rpn_net_file()
        net = caffe.Net(net_file, caffe.TEST)
        os.remove(net_file)
        pred = np.ones([3, 8, 1, 1])
        label = 5. * np.ones([3, 8, 1, 1])
        net.blobs['pred'].data[...] = pred
        net.blobs['label'].data[...] = label
        net.forward()
        net.backward()
        loss = net.blobs['loss'].data[...]
        actual_loss = 30.
        diff = net.blobs['pred'].diff[...]
        actual_diff = -1./2 * np.ones([3, 8, 1, 1])
        self.assertTrue(abs(loss - actual_loss) < EPSILON)
        self.assertTrue((abs(diff - actual_diff) < EPSILON).all())


if __name__ == '__main__':
    unittest.main()
