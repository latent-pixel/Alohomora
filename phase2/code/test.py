import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix

from network.Network import CIFAR10Model, ResNet, ResNext
from misc.MiscUtils import *
from misc.DataUtils import ReadDirNames, ReadLabels


# Don't generate pyc codes
sys.dont_write_bytecode = True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def SetupAll(DataPath):
    DirNamesTest = ReadDirNames(DataPath + 'TxtFiles/DirNamesTest.txt')
    TestLabels = ReadLabels(DataPath + 'TxtFiles/LabelsTest.txt')
    return DirNamesTest, TestLabels


def GenerateTestDataset(DataPath):
    DirNamesTest, TestLabels = SetupAll(DataPath)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    TestImages = []
    Labels = []
    for idx in range(len(DirNamesTest)):
        ImgPath = DataPath + DirNamesTest[idx]
        I1 = Image.open(ImgPath + '.png')
        I1 = transform(I1)
        TestLabel = torch.tensor(int(TestLabels[idx]), dtype=torch.long)
        # Append all the images and labels
        TestImages.append(I1)
        Labels.append(TestLabel)
    return torch.stack(TestImages).to(device), torch.stack(Labels).to(device)
                

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))


def ConfusionMatrix(DataPath):
    LabelsTrue = ReadLabels(DataPath + 'TxtFiles/LabelsTest.txt')
    LabelsPred = ReadLabels(DataPath + 'TxtFiles/LabelsPred.txt')
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    print()
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('\nAccuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')


def TestOperation(ModelArch, ModelPath, TestImages, TestLabels, PredLabelsPath):
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = ResNet().to(device)
    if ModelArch == 'LeNet':
        model = CIFAR10Model().to(device)
    elif ModelArch == 'ResNet':
        pass
    elif ModelArch == 'ResNext':
        model = ResNext(bottleneck_width=4, cardinality=32).to(device)
    elif ModelArch == 'DenseNet':
        model = ResNet().to(device)

    if(not (os.path.isfile(ModelPath))):
        print('ERROR: Model does not exist in ' + ModelPath)
        sys.exit()
    else:
        CheckPoint = torch.load(ModelPath)

    model.load_state_dict(CheckPoint['model_state_dict'])
    print('Model loaded...\n')
    print('Number of parameter groups in this model: {}'.format(len(model.state_dict().items())))
    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    model.eval()

    # Pass the entire batch through the model for inference
    with torch.no_grad():
        predictions = model(TestImages)

    PredSaveFile = open(PredLabelsPath, 'w')
    pbar = tqdm(total=len(TestImages))
    for pred in predictions:
        pred_class = torch.argmax(pred).item()
        PredSaveFile.write(str(pred_class) + '\n')
        pbar.update()

    PredSaveFile.close()
    pbar.close()

       
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelArch', default='ResNet', help='Architecture to use: LeNet/ResNet/ResNext/DenseNet')
    Parser.add_argument('--DataPath', default='./phase2/CIFAR10/', help='Path to the CIFAR10 dataset, Default: phase2/CIFAR10/')
    Parser.add_argument('--ModelPath', default='./phase2/checkpoints/ep25_model.ckpt', help='Path to load latest model from, Default:phase2/checkpoints/ep25_model.ckpt')

    Args = Parser.parse_args()
    ModelArch = Args.ModelArch
    DataPath = Args.DataPath
    ModelPath = Args.ModelPath

    PredLabelsPath = DataPath + './TxtFiles/LabelsPred.txt' # Path to save predicted labels
    TestImages, TestLabels = GenerateTestDataset(DataPath)
    TestOperation(ModelArch, ModelPath, TestImages, TestLabels, PredLabelsPath)

    # Plot Confusion Matrix
    ConfusionMatrix(DataPath)


if __name__ == '__main__':
    main()
 
