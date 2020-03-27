from torch import nn


def conv_relu(i, batch_normalization=False, leaky_relu=False):
    nc = 3
    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 1]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]

    cnn = nn.Sequential()

    n_in = nc if i == 0 else nm[i - 1]
    n_out = nm[i]
    cnn.add_module('conv{0}'.format(i), nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
    if batch_normalization:
        cnn.add_module('batchnorm{0}'.format(i), nn.InstanceNorm2d(n_out, track_running_stats=True))
        # cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
    if leaky_relu:
        cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
    else:
        cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
    return cnn


def make_cnn():
    cnn = nn.Sequential()
    cnn.add_module('convRelu{0}'.format(0), conv_relu(0))
    cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(1), conv_relu(1))
    cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(2), conv_relu(2, True))
    cnn.add_module('convRelu{0}'.format(3), conv_relu(3))
    cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(4), conv_relu(4, True))
    cnn.add_module('convRelu{0}'.format(5), conv_relu(5))
    cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(6), conv_relu(6, True))
    cnn.add_module('pooling{0}'.format(4), nn.MaxPool2d(2, 2))

    return cnn
