import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from sklearn import metrics
import torchvision.transforms as transforms
import ssl
from tqdm import tqdm
import warnings

#----- Specify several stablization checking parameters -----#
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

#----- Specify the class names -----#
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#----- Basic convolution block -----#
class ConvBlock(nn.Module):
    def __init__(self, in_fts, out_fts, k, s, p):
        super(ConvBlock, self).__init__()
        self.basic_Cov = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(k, k), stride=(s, s), padding=(p, p)),
            nn.ReLU()
        )

    def forward(self, input_img):
        x = self.basic_Cov(input_img)
        return x

#----- Reduced convolution block -----#
class ReduceConvBlock(nn.Module):
    def __init__(self, in_fts, out_fts_1, out_fts_2, k, p):
        super(ReduceConvBlock, self).__init__()
        self.reduced_Conv = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts_1, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_fts_1, out_channels=out_fts_2, kernel_size=(k, k), stride=(1, 1), padding=(p, p)),
            nn.ReLU()
        )

    def forward(self, input_img):
        x = self.reduced_Conv(input_img)
        return x

#----- Auxilary classifier block -----#
class AuxClassifier(nn.Module):
    def __init__(self, in_fts, num_classes):
        super(AuxClassifier, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3))
        self.conv = nn.Conv2d(in_channels=in_fts, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(4 * 4 * 128, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.avgpool(input_img)
        x = self.conv(x)
        x = self.relu(x)
        x = x.reshape(N, -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

#----- Inception Module block -----#
class InceptionModule(nn.Module):
    def __init__(self, curr_in_fts, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pool_proj):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvBlock(curr_in_fts, f_1x1, 1, 1, 0)
        self.conv2 = ReduceConvBlock(curr_in_fts, f_3x3_r, f_3x3, 3, 1)
        self.conv3 = ReduceConvBlock(curr_in_fts, f_5x5_r, f_5x5, 5, 2)

        self.pool_projection = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=curr_in_fts, out_channels=f_pool_proj, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )

    def forward(self, input_img):
        out1 = self.conv1(input_img)
        out2 = self.conv2(input_img)
        out3 = self.conv3(input_img)
        out4 = self.pool_projection(input_img)

        x = torch.cat([out1, out2, out3, out4], dim=1)
        return x

#----- Structure of CNN1 -----#
class CNN1(nn.Module):
    def __init__(self, num_class=10):
        super(CNN1, self).__init__()
        in_fts=3
        self.conv1 = ConvBlock(in_fts, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Sequential(
            ConvBlock(64, 64, 1, 1, 0),
            ConvBlock(64, 192, 3, 1, 1)
        )

        self.inception_3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.inception_4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.inception_5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.aux_classifier1 = AuxClassifier(512, num_class)
        self.aux_classifier2 = AuxClassifier(528, num_class)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_class),
        )

    def forward(self, input_img, isTraining):
        self.training = isTraining
        N = input_img.shape[0]
        x = self.conv1(input_img)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool1(x)
        x = self.inception_4a(x)
        out1 = self.aux_classifier1(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        out2 = self.aux_classifier2(x)
        x = self.inception_4e(x)
        x = self.maxpool1(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        if self.training == True:
            return [x, out1, out2]
        else:
            return x

#----- Evaluate the model -----#
def evaluate(testloader, net, cuda, isCheckingOverfitting):
    if isCheckingOverfitting:
        log_filename = 'task3-CNN1-checkOverfitting.log'
    else:
        log_filename = 'task3-CNN1.log'

    correct = 0
    total = 0
    f1_score, i = 0, 0

    #----- Detach gradients during evaluation -----#
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            # predict classes
            outputs = net(images, False)
            # choose the class with the highest prediction value as the class
            _, pred = torch.max(outputs.data, 1)
            # compute f1 scores for current batch
            f1_score += metrics.f1_score(labels.cpu(), pred.cpu(), average="macro")
            i += 1
            with open(log_filename, "a") as file:
                file.write('\n')
                file.write(metrics.classification_report(labels.cpu(), pred.cpu(), target_names=classes))
                file.write('\n')

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    with open(log_filename, "a") as file:
        file.write('\n')
        file.write('overall f1 score {:.4f}\n'.format(f1_score / i))
        file.write('Accuracy of the network on the test set: %d %%\n' % (
                100 * correct / total))
        file.write('\n')

    #----- Count predictions for each class -----#
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    #----- Detach gradients during evaluation -----#
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images, False)
            _, predictions = torch.max(outputs, 1)
            # calculate the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    #----- Write the accuracy for each class into the logfile -----#
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        with open(log_filename, "a") as file:
            file.write("Accuracy for class {:5s} is: {:.1f} %\n".format(classname,
                                                                      accuracy))
    
    if isCheckingOverfitting:
        print('Finish printing evaluation on train data (for overfitting checking).\n')
    else:
        print('Finish printing evaluation on test data.\n')

def main():
    #----- Check whether cuda is available (Train on GPU) -----#
    cuda = torch.cuda.is_available()
    # print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

    # with open('task3-CNN1.log', "w") as file:
    #     file.write('Start Training\n')

    #----- Preprocess dataset, cropped into 224*224*3 -----#
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #----- Load train and test dataset -----#
    batch_size = 64 #!!! May need to change this if the computer GPU's memory is not enough !!!

    print('Checking whether the dataset is downloaded:')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                             shuffle=False, num_workers=0) #!!! May need to change this if the computer GPU's memory is not enough !!!
    print('Finish loading the dataset.\n')

    #----- init model -----#
    net = CNN1(len(classes))
    if cuda:
        net = net.cuda()

    # init loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    #----- start training -----#
    print('Start training:')
    for epoch in range(50):  # loop over the dataset multiple times
        running_loss = 0.0
        with tqdm(trainloader, unit="batch") as tepoch:
            for data in tepoch:
                # get the inputs; data is a list of [inputs, labels]
                tepoch.set_description(f"Epoch {epoch}")
                inputs, labels = data
                if cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs, o1, o2 = net(inputs, True)
                discount = 0.3
                loss = criterion(outputs, labels) + discount*(criterion(o1, labels) + criterion(o2, labels))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

    print('Finished Training\n')

    #----- Save trained model -----#
    print('Start saving models:')
    MODEl_PATH = './CNN_result/CNN1/task3-CNN1.pth'
    torch.save(net.state_dict(), MODEl_PATH)
    print('Model saved at ' + MODEl_PATH + '\n')

    #----- Evaluate model on the test set -----#
    print('Start evaluation on test data')
    net.load_state_dict(torch.load(MODEl_PATH))
    if cuda:
        net.cuda()

    isCheckingOverfitting = False
    evaluate(trainloader, net, cuda, isCheckingOverfitting)

    #----- Evaluate model on the train set(for overfitting checking) -----#
    print('Start evaluation on train data (for overfitting checking)')

    MODEl_PATH = './CNN_result/CNN1/task3-CNN1.pth'
    net.load_state_dict(torch.load(MODEl_PATH))
    if cuda:
        net.cuda()

    batch_size = 1000 #!!! May need to change this if the computer GPU's memory is not enough !!!

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    # evaluate model on the test set
    # write results into the log file
    isCheckingOverfitting = True
    evaluate(trainloader, net, cuda, isCheckingOverfitting)


if __name__ == '__main__':
    main()
