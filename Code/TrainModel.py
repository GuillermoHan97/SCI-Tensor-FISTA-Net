#            Train Model
#           Xiaochen  Han
#            Apr 26 2019
#    guillermo_han97@sjtu.edu.cn
#

import tensorflow as tf
import numpy as np
import DefineParam as DP
import LoadData as LD
import scipy.io as sio
import h5py
import os
from time import time
from PIL import Image
import math
from time import time

# Get param
pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, trainFile, testFile, maskFile, saveDir, modelDir, name = DP.get_param()

outputFile = "log%s.txt" % name

# Train Model
def train_model(sess, saver, costAll, optmAll, Yinput, prediction, trainLabel, trainPhi, Xinput, Xoutput):
    # calculate sum phi for initialization 
    sumPhi = np.divide(1, np.maximum(np.sum(trainPhi, axis=2), 1)).tolist()

    # check log file
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    out = open(outputFile, 'a')
    out.write('%s\n' % name)
    out.write('------------------------------------------------------------------------------------------------------------------------------------------\n')
    out.close()

    # pre-process
    randAll = np.random.permutation(nTrainData // batchSize)
    trainPart = randAll[ :int(trainScale*len(randAll))]
    validatingPart = randAll[int(trainScale*len(randAll)): ]

    # train
    for epoch in range(1, nEpoch + 1):  # all epoch
        # training part
        batchCount = 0
        for batchi in trainPart:                       
            batchCount += 1
            print("training epoch:%d/%d batch:%d/%d, establishing dictionary" % (epoch, nEpoch, batchCount, len(trainPart)))
            xoutput = trainLabel[batchSize*batchi:batchSize*(batchi + 1), :, :, :]                              # label
            yinput = np.sum(np.multiply(xoutput, trainPhi), axis=3)                                             # measurement
            xinput = np.tile(np.reshape(np.multiply(sumPhi, yinput), [-1, pixel, pixel, 1]), [1, 1, 1, nFrame]) # initialization
            yinput = np.reshape(yinput, (-1, pixel, pixel, 1))                                                  # measurement in correct shape
            feedDict = {Xinput: xinput, Xoutput: xoutput, Yinput: yinput}

            print("training epoch:%d/%d batch:%d/%d, optmizing loss function" % (epoch, nEpoch, batchCount, len(trainPart)))
            sess.run(optmAll, feed_dict=feedDict)

        # validating part
        batchCount = 0
        avgPSNR = 0
        for batchi in validatingPart:
            batchCount += 1
            print("validating epoch:%d/%d batch:%d/%d, establishing dictionary" % (epoch, nEpoch, batchCount, len(validatingPart)))
            xoutput = trainLabel[batchSize*batchi:batchSize*(batchi + 1), :, :, :]
            yinput = np.sum(np.multiply(xoutput, trainPhi), axis=3)
            xinput = np.tile(np.reshape(np.multiply(sumPhi, yinput), [-1, pixel, pixel, 1]), [1, 1, 1, nFrame])

            yinput = np.reshape(yinput, (-1, pixel, pixel, 1))
            feedDict = {Xinput: xinput, Xoutput: xoutput, Yinput: yinput}
            start = time()
            result = sess.run(prediction[-1], feed_dict = feedDict)            
            end = time()
            PSNR = psnr(xoutput, result)
            print("validating epoch:%d/%d batch:%d/%d, PSNR: %.4f, time: %.2f" % (epoch, nEpoch, batchCount, len(validatingPart), PSNR, end-start))
            avgPSNR += PSNR

        avgPSNR /= np.maximum(len(validatingPart), 1)
        validateInfo = "Epoch:%d/%d, avg validating PSNR: %.4f\n" % (epoch, nEpoch, avgPSNR)
        print(validateInfo)

        # write info into log file
        out = open(outputFile, 'a')
        out.write(validateInfo)
        out.close()

        # save model
        if epoch % 50 == 0:
            saver.save(sess, '%s/%d.cpkt' % (modelDir, epoch))
            print('model saved\n')

    sess.close()


# calculate psnr
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    return 20*math.log10(1.0/math.sqrt(mse))

