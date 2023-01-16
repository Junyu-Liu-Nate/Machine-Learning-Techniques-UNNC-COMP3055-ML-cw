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

#----- Structure of CNN4 -----#
class CNN4(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN4, self).__init__()
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=1024, out_features=self.num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

#----- Evaluate the model -----#
def evaluate(testloader, net, cuda, isCheckingOverfitting):
    if isCheckingOverfitting:
        log_filename = 'task3-CNN4-checkOverfitting.log'
    else:
        log_filename = 'task3-CNN4.log'

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

    # with open('task3-CNN4.log', "w") as file:
    #     file.write('Start Training')

    #----- preprocess dataset, cropped into 224*224*3 -----#
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #----- Load train and test dataset -----#
    batch_size = 64

    print('Checking whether the dataset is downloaded:')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=0)
    print('Finish loading the dataset.\n')

    #----- init model -----#
    net = CNN4(len(classes))
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
    # saved trained model
    MODEl_PATH = './CNN_result/CNN4/task3-CNN4.pth'
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

    MODEl_PATH = './CNN_result/CNN4/task3-CNN4.pth'
    net.load_state_dict(torch.load(MODEl_PATH))
    if cuda:
        net.cuda()

    batch_size = 500

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    # evaluate model on the train set to check the overfitting status
    # write results into the log file
    isCheckingOverfitting = True
    evaluate(trainloader, net, cuda, isCheckingOverfitting)


if __name__ == '__main__':
    main()
