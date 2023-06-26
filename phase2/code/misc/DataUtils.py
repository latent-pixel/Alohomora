import os
import sys
import random
import numpy as np


# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll(BasePath, CheckPointPath):
    # Setup DirNames
    DirNamesTrain =  SetupDirNames(BasePath)
    # print(DirNamesTrain)

    # Read and Setup Labels
    LabelsPathTrain = BasePath + 'TxtFiles/LabelsTrain.txt'
    TrainLabels = ReadLabels(LabelsPathTrain)

    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(CheckPointPath))):
       os.makedirs(CheckPointPath)
        
    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 100 
    # Number of passes of Val data with MiniBatchSize 
    NumTestRunsPerEpoch = 5
    
    # Image Input Shape
    ImageSize = [32, 32, 3]
    NumTrainSamples = len(DirNamesTrain)

    # Number of classes
    NumClasses = 10

    return DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses


def ReadLabels(LabelsPath):
    if(not (os.path.isfile(LabelsPath))):
        print('ERROR: Train Labels do not exist in '+LabelsPath)
        sys.exit()
    else:
        Labels = open(LabelsPath, 'r')
        Labels = Labels.read()
        Labels = list(map(float, Labels.split()))
    return Labels
    

def SetupDirNames(BasePath): 
    DirNamesTrain = ReadDirNames(BasePath + 'TxtFiles/DirNamesTrain.txt')        
    return DirNamesTrain


def ReadDirNames(ReadPath):
    """
    Inputs: 
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames has the full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames


def TrainValSplit(DirNames, Labels):
    DirNames, Labels = np.array(DirNames), np.array(Labels)
    # Seeding to maintain consistency in Train-Val splitting
    random.seed(42)
    # Randomly selecting 5000 indices
    RandIdx = random.sample(range(len(DirNames)), 5000)
    # Validation Data
    ValDirNames = DirNames[RandIdx]
    ValLabels = Labels[RandIdx]
    # Train Data
    TrainDirNames = np.delete(DirNames, RandIdx)
    TrainLabels = np.delete(Labels, RandIdx)
    return TrainDirNames, ValDirNames, TrainLabels, ValLabels
