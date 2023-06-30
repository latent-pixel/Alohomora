import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW, SGD, lr_scheduler
import sys
import random
from PIL import Image
import argparse
import random
from tqdm import tqdm

from network.Network import CIFAR10Model, ResNet
from misc.MiscUtils import *
from misc.DataUtils import *


# Don't generate pyc codes
sys.dont_write_bytecode = True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def GenerateBatch(DataPath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize):
    I1Batch = []
    LabelBatch = []
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain)-1)
        ImageNum += 1
    
        # Standardization/Data augmentation!
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        ImgPath = DataPath + DirNamesTrain[RandIdx]
        I1 = Image.open(ImgPath + '.png')
        I1 = transform(I1)
        Label = torch.tensor(int(TrainLabels[RandIdx]), dtype=torch.long)

        # Append all the images and labels
        I1Batch.append(I1)
        LabelBatch.append(Label)
        
    return torch.stack(I1Batch).to(device), torch.stack(LabelBatch).to(device)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of training images: ' + str(NumTrainSamples))
    print('Therefore, train-val split: ' + str(NumTrainSamples-0.1*NumTrainSamples) + "-" + str(0.1*NumTrainSamples))
    print('Number of training epochs: ' + str(NumEpochs))
    print('Mini-batch size: ' + str(MiniBatchSize))
    print('Factor of reduction in training data: ' + str(DivTrain))
    
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              
    

def TrainOperation(ModelArch, DataPath, DirNames, Labels, 
                   ImageSize, NumEpochs, MiniBatchSize, DivTrain, 
                   LatestFile, LogsPath, SaveCheckPoint, CheckPointPath):
    # Initialize the model
    model = ResNet().to(device)
    if ModelArch == 'LeNet':
        model = CIFAR10Model().to(device)
    elif ModelArch == 'ResNet':
        model = ResNet().to(device)
    elif ModelArch == 'ResNeXt':
        model = ResNet().to(device)
    elif ModelArch == 'DenseNet':
        model = ResNet().to(device)

    # Splitting training data into training and validation datasets 
    TrainDirNames, ValDirNames, TrainLabels, ValLabels = TrainValSplit(DirNames, Labels)
    NumTrainSamples = len(TrainDirNames)

    StepSize = 5
    Optimizer = SGD(model.parameters(), lr=1e-1, momentum=0.9)
    Scheduler = lr_scheduler.StepLR(Optimizer, step_size=StepSize, gamma=0.2)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + '.ckpt')
        # Extract only numbers from the name
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('\nLoaded latest checkpoint with the name ' + LatestFile + '....\n')
    else:
        StartEpoch = 0
        print('\nNew model initialized....\n')
    
    start_timer = tic()
    for Epoch in range(StartEpoch, NumEpochs):
        print("Epoch [{}]".format(Epoch+1))

        # First step: increase batch size by a factor of 5
        if Epoch == 0:
            MiniBatchSize *= 5
        # Decay learning rate at each subsequent step
        elif Epoch % StepSize == 0:
            Scheduler.step()

        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
        EpochHistory = []

        pbar_inner = tqdm(total=NumIterationsPerEpoch)
        for Iter in range(NumIterationsPerEpoch):
            TrainBatch = GenerateBatch(DataPath, TrainDirNames, TrainLabels, ImageSize, MiniBatchSize)
            ValidationBatch = GenerateBatch(DataPath, ValDirNames, ValLabels, ImageSize, 2*MiniBatchSize)

            Optimizer.zero_grad()

            # Predict output/loss with forward pass
            BatchLoss, BatchAcc = model.training_step(TrainBatch)

            BatchLoss.backward()
            Optimizer.step()
            
            # # Save checkpoint every some SaveCheckPoint's iterations
            # if Iter % SaveCheckPoint == 0:
            #     # Save the Model learnt in this epoch
            #     SaveName =  CheckPointPath + str(Epoch) + 'a' + str(Iter) + 'model.ckpt'
            #     torch.save({'epoch': Epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
            #     # print('\n' + SaveName + ' Model Saved...')

            result = model.validation_step(ValidationBatch)
            result.update({'train_acc': BatchAcc})
            EpochHistory.append(result)

            pbar_inner.update()

        pbar_inner.close()
        result = model.epoch_end(EpochHistory)  # calculates the loss and acc avgs
        model.fetch_epoch_results(result)  # prints the epoch loss and accuracy

        # Update Tensorboard
        Writer.add_scalar(f'Accuracy/TrainAccuracy', result['train_acc'], Epoch)
        Writer.add_scalar(f'Accuracy/ValAccuracy', result["val_acc"], Epoch)
        Writer.add_scalar('ValLoss', result["val_loss"], Epoch)
        Writer.flush()  # Without flushing, the tensorboard doesn't get updated until a lot of iterations!

        # Save model every epoch
        SaveName = CheckPointPath + 'ep' + str(Epoch+1) + '_model.ckpt'
        torch.save({'epoch': Epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': BatchLoss}, SaveName)
        print('Model saved at ' + SaveName + '\n')

    training_time = toc(start_timer)
    print("The total time taken to train the model: {} seconds".format(round(training_time, 2)))
  

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the training process
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelArch', default='ResNet', help='Architecture to use: LeNet/ResNet/ResNeXt/DenseNet')
    Parser.add_argument('--DataPath', default='./phase2/CIFAR10/', help='Path to the CIFAR10 dataset, Default: phase2/CIFAR10/')
    Parser.add_argument('--CheckPointPath', default='./phase2/checkpoints/', help='Path to save Checkpoints, Default: phase2/checkpoints/')
    Parser.add_argument('--LogsPath', default='./phase2/logs/', help='Path to save Logs for Tensorboard, Default=phase2/logs/')
    Parser.add_argument('--NumEpochs', type=int, default=5, help='Number of Epochs to Train for, Default:5')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=128, help='Size of the MiniBatch to use, Default:32')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')

    Args = Parser.parse_args()
    ModelArch = Args.ModelArch
    DataPath = Args.DataPath
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    
    # Setup all needed parameters including file reading
    DirNames, SaveCheckPoint, ImageSize, NumTrainSamples, Labels, NumClasses = SetupAll(DataPath, CheckPointPath)
    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)
    TrainOperation(ModelArch, DataPath, DirNames, Labels, 
                   ImageSize, NumEpochs, MiniBatchSize, DivTrain, 
                   LatestFile, LogsPath, SaveCheckPoint, CheckPointPath)

    
if __name__ == '__main__':
    main()
