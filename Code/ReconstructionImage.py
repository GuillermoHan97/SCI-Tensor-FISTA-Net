#        Reconstruction Image
#           Xiaochen  Han
#            Apr 26 2019
#    guillermo_han97@sjtu.edu.cn
#

import scipy.io as sio
import numpy as np
from time import time
from PIL import Image
import math
import LoadData as LD
import DefineParam as DP
import os
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

# get param
pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, trainFile, testFile, maskFile, saveDir, modelDir, name = DP.get_param()

# load and form test image
def reconstruct_image(sess, Yinput, prediction, Xinput, Xoutput, testLabel, testPhi):
    # for initialization
    sumPhi = np.divide(1, np.maximum(np.sum(testPhi, axis=2), 1)).tolist()

    # check reconstructed images saving dir
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # pre-process
    xoutput = testLabel                                                                                    # labels
    yinput = np.sum(np.multiply(xoutput, testPhi), axis=3)                                                 # measurement
    xinput =np.tile(np.reshape(np.multiply(yinput, sumPhi), [-1, pixel, pixel, 1]), [1, 1, 1, nFrame])     # initialization
    yinput = np.reshape(yinput, (-1, pixel, pixel, 1))                                                     # measurement in correct shape
    feedDict = {Xinput: xinput, Xoutput: xoutput, Yinput: yinput}
    init = xinput

    # do reconstruction
    start = time()
    rec = sess.run(prediction[-1], feed_dict=feedDict)
    end = time()

    # calculate psnr
    PSNR = psnr(xoutput, rec)
    recInfo = "Rec avg PSNR %.4f time= %.2fs\n" % (PSNR, (end - start))
    print(recInfo)

    # output reconstruction image
    for i in range(4):
        for j in range(nFrame):
            PSNR = psnr(rec[i, :, :, j], xoutput[i, :, :, j])
            print("Frame %d, PSNR: %.2f" % (i*nFrame+j, PSNR))
            outImg = np.hstack((xoutput[i, :, :, j], rec[i, :, :, j]))
            imgRecName = "%s/frame%d phase%d ncpkt%d PSNR%.2f.png" % (saveDir, i*nFrame+j, nPhase, ncpkt, PSNR)
            imgRec = Image.fromarray(np.clip(255*outImg, 0, 255).astype(np.uint8))
            imgRec.save(imgRecName)

    sess.close()


# calculate psnr
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    return 20*math.log10(1.0/math.sqrt(mse))
