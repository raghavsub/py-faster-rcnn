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
        netFile = robust_l1_loss_layer_net_file()
        self.net = caffe.Net(netFile, caffe.TEST)
        os.remove(netFile)
    
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
    
    def test_close(self):
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


if __name__ == '__main__':
    unittest.main()
