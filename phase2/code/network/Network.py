import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def loss_fn(out, labels):
    loss = nn.CrossEntropyLoss()(out, labels)
    return loss


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)  # Generate predictions
        loss = loss_fn(out, labels)  # Calculate loss
        acc = accuracy(out, labels)
        return loss, acc
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)  # Generate predictions
        val_loss = loss_fn(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': val_loss.detach(), 'val_acc': acc}
        
    def epoch_end(self, outputs):
        train_accs = [x['train_acc'] for x in outputs]  # Combine accuracies
        train_epoch_acc = torch.stack(train_accs).mean()
        val_accs = [x['val_acc'] for x in outputs]
        val_epoch_acc = torch.stack(val_accs).mean()  
        batch_losses = [x['val_loss'] for x in outputs]
        val_epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        return {'train_acc': train_epoch_acc.item(), 'val_acc': val_epoch_acc.item(), 'val_loss': val_epoch_loss.item()}
    
    def fetch_epoch_results(self, result):
        print("TrainAcc: {:.4f}, ValAcc: {:.4f}, ValLoss: {:.4f}\n".format(result['train_acc'], result['val_acc'], result['val_loss']))


class CIFAR10Model(ImageClassificationBase):
    def __init__(self):
        """
        Inputs: 
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        x = self.pool(F.relu(self.conv1(xb)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

