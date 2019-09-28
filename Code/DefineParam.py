#          Define all param
#           Xiaochen  Han
#            May 21 2019
#    guillermo_han97@sjtu.edu.cn
#

# define all param
def get_param():    
    pixel = 256
    batchSize = 8
    nPhase = 5
    nTrainData = 357
    trainScale = 0.9           # scale of training part and validating part
    learningRate = 0.0001
    nEpoch = 50
    nFrame = 8
    ncpkt = nEpoch
    name = 'Vehicle'

    trainFile = './trainData/train%s%d.mat' % (name, pixel)
    testFile = './testData/test%s%d.mat' % (name, pixel)
    maskFile = './maskData/mask%d.mat' % pixel
    saveDir = './recImg/recImg%d' % pixel
    modelDir = './Model%d' % pixel

    return pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, trainFile, testFile, maskFile, saveDir, modelDir, name
