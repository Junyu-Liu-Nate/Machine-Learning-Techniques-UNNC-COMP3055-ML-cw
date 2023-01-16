import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from sklearn import metrics
import torchvision.transforms as transforms
import ssl
from tqdm import tqdm
import warnings
from collections import OrderedDict

#----- Specify several stablization checking parameters -----#
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

#----- Specify the class names -----#
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#----- Depth_wise convolution block -----#
class Depthwise_Conv(nn.Module):
    def __init__(self, in_fts, stride=(1, 1)):
        super(Depthwise_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, in_fts, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=in_fts),
            nn.BatchNorm2d(in_fts),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x

#----- Point_wise convolution block -----#
class Pointwise_Conv(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(Pointwise_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, out_fts, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_fts),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x

#----- Depth_wise seperable convolution block -----#
class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_fts, out_fts, stride=(1, 1)):
        super(Depthwise_Separable_Conv, self).__init__()
        self.depth_wise = Depthwise_Conv(in_fts=in_fts, stride=stride)
        self.point_wise = Pointwise_Conv(in_fts=in_fts, out_fts=out_fts)

    def forward(self, input_image):
        x = self.point_wise(self.depth_wise(input_image))
        return x

#----- Structure of CNN2 -----#
class CNN2(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN2, self).__init__()
        in_fts=3
        num_filter=32

        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, num_filter, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(inplace=True)
        )

        self.in_fts = num_filter

        # If the type of sublist is list: set stride=(2,2)
        # if length of the sublist = 1: set stride=(2,2)
        # if length of the sublist = 2: set means (num_times, num_filter)
        self.nlayer_filter = [
            num_filter * 2,
            [num_filter * pow(2, 2)],
            num_filter * pow(2, 2),
            [num_filter * pow(2, 3)],
            num_filter * pow(2, 3),
            [num_filter * pow(2, 4)],
            [5, num_filter * pow(2, 4)],
            [num_filter * pow(2, 5)],
            num_filter * pow(2, 5)
        ]

        self.DSC = self.layer_construct()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, input_image):
        N = input_image.shape[0]
        x = self.conv(input_image)
        x = self.DSC(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        return x

    # Construct the layer according to the self.nlayer_filter
    def layer_construct(self):
        block = OrderedDict()
        index = 1
        for l in self.nlayer_filter:
            if type(l) == list:
                if len(l) == 2: 
                    for _ in range(l[0]):
                        block[str(index)] = Depthwise_Separable_Conv(self.in_fts, l[1])
                        index += 1
                else: 
                    block[str(index)] = Depthwise_Separable_Conv(self.in_fts, l[0], stride=(2, 2))
                    self.in_fts = l[0]
                    index += 1
            else:
                block[str(index)] = Depthwise_Separable_Conv(self.in_fts, l)
                self.in_fts = l
                index += 1

        return nn.Sequential(block)

#----- Evaluate the model -----#
def evaluate(testloader, net, cuda, isCheckingOverfitting):
    if isCheckingOverfitting:
        log_filename = 'task3-CNN2-checkOverfitting.log'
    else:
        log_filename = 'task3-CNN2.log'

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
            outputs = net(images)
            # choose the class with the highest prediction value as the class
            _, pred = torch.max(outputs.data, 1)
            # compute f1 scores for current batch
            f1_score += metrics.f1_score(labels.cpu(), pred.cpu(), average="macro")
            i += 1
            with open(log_filename, "a") as file:
                file.write(metrics.classification_report(labels.cpu(), pred.cpu(), target_names=classes))
                file.write('\n')

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    #----- Write the accuracy for each class into the logfile -----#
    with open(log_filename, "a") as file:
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
            outputs = net(images)
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

    # with open('task3-CNN2.log', "w") as file:
    #     file.write('Start Training')

    #----- preprocess dataset, cropped into 224*224*3 -----#
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=0) #!!! May need to change this if the computer GPU's memory is not enough !!!
    print('Finish loading the dataset.\n')

    #----- init model -----#
    net = CNN2(len(classes))
    if cuda:
        net = net.cuda()

    #----- init loss function and optimizer -----#
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
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

    print('Finished Training\n')

    #----- Save trained model -----#
    MODEl_PATH = './CNN_result/CNN2/task3-CNN2.pth'
    torch.save(net.state_dict(), MODEl_PATH)
    print('Model saved at ' + MODEl_PATH + '\n')

    #----- Evaluate model on the test set -----#
    print('Start evaluation on test data')
    net.load_state_dict(torch.load(MODEl_PATH))
    if cuda:
        net.cuda()

    # evaluate model on the test set
    # write results into the log file
    isCheckingOverfitting = True
    evaluate(testloader, net, cuda, isCheckingOverfitting)

    #----- Evaluate model on the train set(for overfitting checking) -----#
    print('Start evaluation on train data (for overfitting checking)')

    MODEl_PATH = './CNN_result/CNN2/task3-CNN2.pth'
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
