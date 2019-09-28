#       Reconstruction of SCI
#           Xiaochen  Han
#            Apr 26 2019#
#    guillermo_han97@sjtu.edu.cn
#

import LoadData as LD
import BuildModel as BM
import TrainModel as TM
import ReconstructionImage as RI
import DefineParam as DP
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

# get param
pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, trainFile, testFile, maskFile, saveDir, modelDir, name = DP.get_param()

# load data
print('-------------------------------------\nLoading Data...\n-------------------------------------\n')
testLabel, testPhi = LD.load_test_data(mat73=False)

# build model
print('-------------------------------------\nBuilding and Restoring Model...\n-------------------------------------\n')
sess, saver, Xinput, Xoutput, Yinput, prediction = BM.build_model(testPhi, restore=True)

# reconstruct image
print('-------------------------------------\nReconstructing Image...\n-------------------------------------\n')
RI.reconstruct_image(sess, Yinput, prediction, Xinput, Xoutput, testLabel, testPhi)

print('-------------------------------------\nReconstructing Accomplished.\n-------------------------------------\n')





