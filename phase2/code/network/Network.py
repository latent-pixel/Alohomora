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
        print("TrainAcc: {:.4f}, ValAcc: {:.4f}, ValLoss: {:.4f}".format(result['train_acc'], result['val_acc'], result['val_loss']))


class CIFAR10Model(ImageClassificationBase):
    def __init__(self):
        """
        Inputs: 
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.bnorm_conv1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bnorm_conv2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.bnorm_fc1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bnorm_fc2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        x = self.pool(F.relu(self.bnorm_conv1(self.conv1(xb))))
        x = self.pool(F.relu(self.bnorm_conv2(self.conv2(x))))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.bnorm_fc1(self.fc1(x)))
        x = F.relu(self.bnorm_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x


######################################################################################
# ResNet20 implementation from "Deep Residual Learning for Image Recognition" #
# https://arxiv.org/abs/1512.03385 #
######################################################################################
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels))
        # Using option (B) from the paper for shortcut connections: projection along with identity mapping
        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                                          nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(residual)
        return F.relu(out)
    

class ResNet(ImageClassificationBase):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU())
        self.res1 = nn.Sequential(ResNetBlock(16, 16, 1),
                                       ResNetBlock(16, 16),
                                       ResNetBlock(16, 16))
        self.res2 = nn.Sequential(ResNetBlock(16, 32, 2),
                                       ResNetBlock(32, 32),
                                       ResNetBlock(32, 32))
        self.res3 = nn.Sequential(ResNetBlock(32, 64, 2),
                                       ResNetBlock(64, 64),
                                       ResNetBlock(64, 64))
        self.glob_avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.conv0(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.glob_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


################################################################################################
# ResNeXt29 implementation from "Aggregated Residual Transformations for Deep Neural Networks" #
# https://arxiv.org/abs/1611.05431 #
################################################################################################
class ResNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d=64, C=1, stride=1) -> None:
        super().__init__()
        group_width = d*C
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, group_width, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(group_width),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(group_width, group_width, 3, stride=stride, padding=1, groups=C),
                                   nn.BatchNorm2d(group_width),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(group_width, out_channels, 1, stride=1),
                                   nn.BatchNorm2d(out_channels),
                                   )
        # Using option (B) from the paper for shortcut connections: projection along with identity mapping
        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                                          nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(residual)
        return F.relu(out)
    

class ResNext(ImageClassificationBase):
    def __init__(self, bottleneck_width=64, cardinality=1) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.stage1 = nn.Sequential(ResNextBlock(64, 256, d=bottleneck_width, C=cardinality, stride=1),
                                  ResNextBlock(256, 256, d=bottleneck_width, C=cardinality),
                                  ResNextBlock(256, 256, d=bottleneck_width, C=cardinality))
        self.stage2 = nn.Sequential(ResNextBlock(256, 512, d=2*bottleneck_width, C=cardinality, stride=2),
                                  ResNextBlock(512, 512, d=2*bottleneck_width, C=cardinality),
                                  ResNextBlock(512, 512, d=2*bottleneck_width, C=cardinality))
        self.stage3 = nn.Sequential(ResNextBlock(512, 1024, d=4*bottleneck_width, C=cardinality, stride=2),
                                  ResNextBlock(1024, 1024, d=4*bottleneck_width, C=cardinality),
                                  ResNextBlock(1024, 1024, d=4*bottleneck_width, C=cardinality))
        self.glob_avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.conv0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.glob_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


######################################################################################
# DenseNet40 implementation from "Densely Connected Convolutional Networks" #
# https://arxiv.org/abs/1608.06993 #
######################################################################################
class BasicBlock(nn.Module):
    # k is the growth rate parameter
    def __init__(self, in_channels, k=12) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.BatchNorm2d(in_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels, k, kernel_size=3, padding=1))
    
    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), dim=1)
        return out


class DenseNet(ImageClassificationBase):
    def __init__(self, k=12, block_sizes=[6, 6, 6]) -> None:
        super().__init__()
        self.k = k
        self.in_channels = 16
        self.conv0 = nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1)
        self.dense_block1 = self._make_dense_block(block_sizes[0])
        self.trans_block1 = self._make_trans_block(self.in_channels)
        self.dense_block2 = self._make_dense_block(block_sizes[1])
        self.trans_block2 = self._make_trans_block(self.in_channels)
        self.dense_block3 = self._make_dense_block(block_sizes[2])
        
        self.glob_pool = nn.Sequential(nn.BatchNorm2d(self.in_channels),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(self.in_channels, 10)

    def _make_dense_block(self, block_size):
        block_list = []
        for i in range(block_size):
            block_list.append(BasicBlock(self.in_channels, self.k))
            self.in_channels += self.k
        block_list = nn.Sequential(*block_list)
        return block_list
    
    def _make_trans_block(self, in_channels):
        block = nn.Sequential(nn.BatchNorm2d(in_channels),
                            nn.ReLU(),
                            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
                            nn.AvgPool2d(kernel_size=2, stride=2))
        self.in_channels = in_channels // 2
        return block
    
    def forward(self, x):
        out = self.conv0(x)
        out = self.trans_block1(self.dense_block1(out))
        out = self.trans_block2(self.dense_block2(out))
        out = self.dense_block3(out)
        out = self.glob_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
