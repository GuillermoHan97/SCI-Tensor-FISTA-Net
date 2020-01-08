#             Load data
#           Xiaochen  Han
#            Apr 26 2019
#    guillermo_han97@sjtu.edu.cn
#

import tensorflow as tf
import scipy.io as sio
import numpy as np
import DefineParam as DP
import h5py


# Get param
pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, trainFile, testFile, maskFile, saveDir, modelDir, name = DP.get_param()


# Load training data
def load_train_data(mat73=False):
    if mat73 == True:                                                # if .mat file is too big, use h5py to load
        trainData = h5py.File(trainFile)
        trainLabel = np.transpose(trainData['labels'], [3, 2, 1, 0])
    else:
        trainData = sio.loadmat(trainFile)
        trainLabel = trainData['labels']                             # labels

    maskData = sio.loadmat(maskFile)
    phi = maskData['phi']                                            # mask

    del trainData, maskData
    return trainLabel, phi


# Load testing data
def load_test_data(mat73=False):
    if mat73 == True:
        testData = h5py.File(testFile)
        testLabel = np.transpose(testData['labels'], [3, 2, 1, 0])
    else:
        testData = sio.loadmat(testFile)
        testLabel = testData['labels']
    
    maskData = sio.loadmat(maskFile)
    phi = maskData['phi']

    del testData, maskData
    return testLabel, phi


# Compute essential variables
def pre_calculate(phi):
    Xinput = tf.placeholder(tf.float32, [None, pixel, pixel, nFrame])      # X0
    Xoutput = tf.placeholder(tf.float32, [None, pixel, pixel, nFrame])     # labels
    Yinput = tf.placeholder(tf.float32, [None, pixel, pixel, 1])           # measurement
    Phi = tf.constant(phi)
    PhiT = Phi

    return Xinput, Xoutput, Phi, PhiT, Yinput













