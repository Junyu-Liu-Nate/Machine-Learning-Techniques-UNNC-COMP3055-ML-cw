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

#----- Structure of CNN3 -----#
class CNN3(nn.Module):
    def __init__(self, num_classes):
        super(CNN3, self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=5, stride=(4, 4), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=(2, 2)),
            nn.Softmax(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.Softmax(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 9216)
        x = self.classifier(x)
        return x

#----- Evaluate the model -----#
def evaluate(testloader, net, cuda, isCheckingOverfitting):
    if isCheckingOverfitting:
        log_filename = 'task3-CNN3-checkOverfitting.log'
    else:
        log_filename = 'task3-CNN3.log'

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

    # with open('task3-CNN3.log', "w") as file:
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
    net = CNN3(len(classes))
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
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

    print('Finished Training\n')

    #----- Save trained model -----#
    print('Start saving models:')
    MODEl_PATH = './CNN_result/CNN3/task3-CNN3.pth.pth'
    torch.save(net.state_dict(), MODEl_PATH)
    print('Model saved at ' + MODEl_PATH + '\n')

    #----- Evaluate model on the test set -----#
    print('Start evaluation on test data')
    net.load_state_dict(torch.load(MODEl_PATH))
    if cuda:
        net.cuda()

    # write results into the log file
    isCheckingOverfitting = False
    evaluate(testloader, net, cuda, isCheckingOverfitting)

    #----- Evaluate model on the train set(for overfitting checking) -----#
    print('Start evaluation on train data (for overfitting checking)')

    MODEl_PATH = './CNN_result/CNN3/task3-CNN3.pth'
    net.load_state_dict(torch.load(MODEl_PATH))
    if cuda:
        net.cuda()

    batch_size = 1000 #!!! May need to change this if the computer GPU's memory is not enough !!!
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    # write results into the log file
    isCheckingOverfitting = True
    evaluate(trainloader, net, cuda, isCheckingOverfitting)


if __name__ == '__main__':
    main()
