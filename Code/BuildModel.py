#         Build train Model
#           Xiaochen  Han
#            Apr 26 2019
#    guillermo_han97@sjtu.edu.cn
#

import tensorflow as tf
import numpy as np
import LoadData as LD
import DefineParam as DP
import tensorflow_probability as tfp
from tensorflow.contrib.layers.python.layers import batch_norm

# Get param
pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, trainFile, testFile, maskFile, saveDir, modelDir, name = DP.get_param()

# Build Model
def build_model(phi, restore=False):
    # pre-process
    Xinput, Xoutput, Phi, PhiT, Yinput = LD.pre_calculate(phi)

    # build model
    prediction, predictionSymmetric, transField = build_fista(Xinput, Phi, PhiT, Yinput, reuse=False)

    # loss function
    costMean, costSymmetric, costSparsity = compute_cost(prediction, predictionSymmetric, Xoutput, transField)
    costAll = costMean + 0.01*costSymmetric + 0.001*costSparsity
    optmAll = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(costAll)

    # set up
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    sess = tf.Session(config=config)

    if restore is False:                                       # training
        sess.run(init)
        return sess, saver, Xinput, Xoutput, costAll, optmAll, Yinput, prediction
    else:                                                      # reconstruction
        saver.restore(sess, '%s/%d.cpkt' % (modelDir, ncpkt))
        return sess, saver, Xinput, Xoutput, Yinput, prediction        


# Add weight
def get_filter(wShape, nOrder):
    weight = tf.get_variable(shape=wShape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='weight_%d' % (nOrder))
    return weight


def build_one_phase(layerxk, layerzk, Phi, PhiT, Yinput):
    # params
    lambdaStep = tf.Variable(0.1, dtype=tf.float32)
    softThr = tf.Variable(0.1, dtype=tf.float32)
    t = tf.Variable(1, dtype=tf.float32)
    convSize1 = 64
    convSize2 = 64
    convSize3 = 64
    filterSize1 = 3
    filterSize2 = 3
    filterSize3 = 3 

    # get rk from zk
    rk = tf.reduce_sum(tf.multiply(Phi, layerzk[-1]), axis=3)
    rk = tf.reshape(rk, shape=[-1, pixel, pixel, 1])
    rk = tf.subtract(rk, Yinput)
    rk = tf.multiply(PhiT, tf.tile(rk, [1, 1, 1, nFrame]))
    rk = tf.scalar_mul(lambdaStep, rk)
    rk = tf.subtract(layerzk[-1], rk)

    # F(rk)
    weight0 = get_filter([filterSize1, filterSize1, nFrame, convSize1], 0)
    weight11 = get_filter([filterSize2, filterSize2, convSize1, convSize2], 11)
    weight12 = get_filter([filterSize3, filterSize3, convSize2, convSize3], 12)
    Frk = tf.nn.conv2d(rk, weight0, strides=[1, 1, 1, 1], padding='SAME')
    tmp = Frk
    Frk = tf.nn.conv2d(Frk, weight11, strides=[1, 1, 1, 1], padding='SAME')
    Frk = tf.nn.relu(Frk)
    Frk = tf.nn.conv2d(Frk, weight12, strides=[1, 1, 1, 1], padding='SAME')

    # soft threshold, soft(F(rk), softThr)
    softFrk = tf.multiply(tf.sign(Frk), tf.nn.relu(tf.subtract(tf.abs(Frk), softThr)))    

    # ~F(soft(F(rk), softThr))
    weight13 = get_filter([filterSize3, filterSize3, convSize3, convSize2], 53)
    weight14 = get_filter([filterSize2, filterSize2, convSize2, convSize1], 54)
    weight6 = get_filter([filterSize1, filterSize1, convSize1, nFrame], 6)
    FFrk = tf.nn.conv2d(softFrk, weight13, strides=[1, 1, 1, 1], padding='SAME')
    FFrk = tf.nn.relu(FFrk)
    FFrk = tf.nn.conv2d(FFrk, weight14, strides=[1, 1, 1, 1], padding='SAME')
    FFrk = tf.nn.conv2d(FFrk, weight6, strides=[1, 1, 1, 1], padding='SAME')

    # xk = rk + ~F(soft(F(rk), softThr))
    xk = tf.add(rk, FFrk)
    zk = (1 + t)*xk - t*layerxk[-1]

    # Symmetric constraint
    sFFrk = tf.nn.conv2d(Frk, weight13, strides=[1, 1, 1, 1], padding='SAME')
    sFFrk = tf.nn.relu(sFFrk)
    sFFrk = tf.nn.conv2d(sFFrk, weight14, strides=[1, 1, 1, 1], padding='SAME')
    symmetric = sFFrk - tmp
    return xk, zk, symmetric, Frk


# compute fista once (one epoch)
def build_fista(Xinput, Phi, PhiT, Yinput, reuse):
    layerxk = []
    layerzk = []
    layerSymmetric = []
    transField = []
    layerxk.append(Xinput)                            # x0 = x0
    layerzk.append(Xinput)                            # z0 = x0
    for i in range(nPhase):
        with tf.variable_scope('conv_%d' % (i), reuse=reuse):
            xk, zk, convSymmetric, field = build_one_phase(layerxk, layerzk, Phi, PhiT, Yinput)
            layerxk.append(xk)
            layerzk.append(zk)
            layerSymmetric.append(convSymmetric)
            transField.append(field)
    return layerxk, layerSymmetric, transField


# compute loss function
def compute_cost(prediction, predictionSymmetric, Xoutput, transField):
    costMean = 0
    costMean += tf.reduce_mean(tf.square(prediction[-1] - Xoutput))
    costSymmetric = 0
    for k in range(nPhase):
        costSymmetric += tf.reduce_mean(tf.square(predictionSymmetric[k]))
    costSparsity = 0
    for k in range(nPhase):
        costSparsity += tf.reduce_mean(tf.abs(transField[k]))
    return costMean, costSymmetric, costSparsity

