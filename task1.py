import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

#----- Specify pytorch/dataset parameters -----#
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

#----- Download/load the dataset -----#
print("----------- Checking whether the dataset is downloaded --------------")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
print("")

#----- Obtain one batch of train data and related labels -----#
traindata = iter(trainloader)
train_images, train_labels = next(traindata)
train_images_flat = train_images.reshape(-1,3072)

#----- obtain one batch of test data and related labels -----#
testdata = iter(testloader)
test_images, test_labels = next(testdata)
test_images_flat = test_images.reshape(-1,3072)

#----- Use PCA class for dimension reduction -----#
# The `n_components` parameter controls the different reduction of features, `n_components` is set in {1, 2, 4, 8, 16}
print("----------- Using PCA with different number of components --------------")
para_list = [1, 2, 4, 8, 16]
for para in para_list:
    print("-------------------------")
    print("n_components: ", para)
    pca = PCA(n_components=para)
    x_train_trans = pca.fit_transform(train_images_flat[:64])
    x_test_trans = pca.transform(test_images_flat[:64])
    print('Information kept: ', sum(pca.explained_variance_ratio_)*100, '%')
    print('Noise variance: ', pca.noise_variance_)

print("\n----------- Task 1 finished --------------")