import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import os

#----- Specify the class names -----#
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#----- Function to print the train statistics into files -----#
def print_train_stats(output_fileName, scores):
    with open(output_fileName, 'w') as fileHandler:
        fileHandler.write('# SVM stats\n\n')

        fileHandler.write('#----- Cross validation status -----#\n')

        fileHandler.write('# fit_time\n')
        for fit_time in scores['fit_time']:
            fileHandler.write(str(fit_time))
            fileHandler.write(' ')
        fileHandler.write('\n\n')

        fileHandler.write('# score_time\n')
        for score_time in scores['score_time']:
            fileHandler.write(str(score_time))
            fileHandler.write(' ')
        fileHandler.write('\n\n')

        fileHandler.write('# test_precision_macro\n')
        for test_precision_macro in scores['test_precision_macro']:
            fileHandler.write(str(test_precision_macro))
            fileHandler.write(' ')
        fileHandler.write('\n\n')

        fileHandler.write('# test_recall_macro\n')
        for test_recall_macro in scores['test_recall_macro']:
            fileHandler.write(str(test_recall_macro))
            fileHandler.write(' ')
        fileHandler.write('\n\n')

        fileHandler.write('# test_f1_macro\n')
        for test_f1_macro in scores['test_f1_macro']:
            fileHandler.write(str(test_f1_macro))
            fileHandler.write(' ')
        fileHandler.write('\n\n')

#----- Function to print the test statistics into files -----#
def print_test_stats(output_fileName, labels, pred, test_precision, test_recall, test_f1_score, test_accuracy):
    with open(output_fileName, 'a') as fileHandler:
        fileHandler.write('#----- Test results -----#\n')

        fileHandler.write(metrics.classification_report(labels, pred, target_names=classes))
        fileHandler.write('\n')

        fileHandler.write('# Overall precision is: ')
        fileHandler.write(str(test_precision))
        fileHandler.write('\n')

        fileHandler.write('# Overall recall is: ')
        fileHandler.write(str(test_recall))
        fileHandler.write('\n')

        fileHandler.write('# Overall f1 score is: ')
        fileHandler.write(str(test_f1_score))
        fileHandler.write('\n')

        fileHandler.write('# Overall accuracy is: ')
        fileHandler.write(str(test_accuracy))
        fileHandler.write('\n')
                     
#----- Specify pytorch/dataset parameters -----#
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 50000 #!!! This can be changed to smaller number to speed up the train !!!

#----- Download/load the dataset -----#
print("----------- Checking whether the dataset is downloaded --------------")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=0)
print("")

#----- Obtain one batch of train data and related labels -----#
traindata = iter(trainloader)
train_images, train_labels = next(traindata)
train_images_flat = train_images.reshape(-1,3072)

#----- Obtain one batch of test data and related labels -----#
testdata = iter(testloader)
test_images, test_labels = next(testdata)
test_images_flat = test_images.reshape(-1,3072)

#----- Train and test SVM classifier -----#
print("----------- Training SVM with different PCA values --------------")

# SVM and PCA parameters
c_list = [0.1, 1, 10, 100, 1000]
pca_para_list = ['1','2','4','8','16']
scoring = ['precision_macro', 'recall_macro', 'f1_macro']

linear_folder_name = 'SVM_result/linear/'
if not os.path.exists('SVM_result/linear/'):
    os.makedirs('SVM_result/linear/')
rbf_folder_name = 'SVM_result/rbf/'
if not os.path.exists('SVM_result/rbf/'):
    os.makedirs('SVM_result/rbf/')

# SVM with linear kernel
print("----- Using Linear kernal -----")
for c in c_list:
    # SVM on the original data
    print("Starting training with no PCA applied:")
    # Train SVM
    clf = svm.SVC(kernel='linear', C=c)
    output = cross_validate(clf, train_images_flat, train_labels, cv=5, scoring=scoring, return_estimator=True)
    
    cv_score_list = output['test_precision_macro'].tolist()
    trained_clf_list = output['estimator']
    select_clf_index = cv_score_list.index(max(cv_score_list))
    trained_clf = trained_clf_list[select_clf_index]
    
    # Test SVM
    test_predict = trained_clf.predict(test_images_flat)
    test_precision = precision_score(test_labels, test_predict, average='macro')
    test_recall = recall_score(test_labels, test_predict, average='macro')
    test_f1_score = f1_score(test_labels, test_predict, average='macro')
    test_accuracy = accuracy_score(test_labels, test_predict)

    file_name = linear_folder_name + 'C_' + str(c) + '_no_PCA' + '.txt'
    print("Printing cross validation results to file: " + file_name)
    print_train_stats(file_name, output)
    print("Printing test results to file: " + file_name)
    print_test_stats(file_name, test_labels, test_predict, test_precision, test_recall, test_f1_score, test_accuracy)
    print('')

    # SVM on data which has different level of dimension reduction from PCA
    for pca_para in pca_para_list:
        print("Starting trainging with PCA of n_components: ", int(pca_para))
        pca = PCA(n_components=int(pca_para))

        x_train_trans = pca.fit_transform(train_images_flat[:50000])
        x_test_trans = pca.transform(test_images_flat[:10000])
        print('Information kept: ', sum(pca.explained_variance_ratio_)*100, '%')
        print('Noise variance: ', pca.noise_variance_)

        # Train SVM
        clf = svm.SVC(kernel='linear', C=c)
        output = cross_validate(clf, x_train_trans, train_labels, cv=5, scoring=scoring, return_estimator=True)
        
        cv_score_list = output['test_precision_macro'].tolist()
        trained_clf_list = output['estimator']
        select_clf_index = cv_score_list.index(max(cv_score_list))
        trained_clf = trained_clf_list[select_clf_index]
        
        # Test SVM
        test_predict = trained_clf.predict(x_test_trans)

        test_precision = precision_score(test_labels, test_predict, average='macro')
        test_recall = recall_score(test_labels, test_predict, average='macro')
        test_f1_score = f1_score(test_labels, test_predict, average='macro')
        test_accuracy = accuracy_score(test_labels, test_predict)

        file_name = linear_folder_name + 'C_' + str(c) + '_PCA_' + pca_para + '.txt'
        print("Printing cross validation results to file: " + file_name)
        print_train_stats(file_name, output)
        print("Printing test results to file: " + file_name)
        print_test_stats(file_name, test_labels, test_predict, test_precision, test_recall, test_f1_score, test_accuracy)
        print('')

# SVM with RBF kernel
print("----- Using rbf kernal -----")
for c in c_list:
    # SVM on the original data
    print("Starting training with no PCA applied:")
    # Train SVM
    clf = svm.SVC(kernel='rbf', C=c)
    output = cross_validate(clf, train_images_flat, train_labels, cv=5, scoring=scoring, return_estimator=True)
    
    cv_score_list = output['test_precision_macro'].tolist()
    trained_clf_list = output['estimator']
    select_clf_index = cv_score_list.index(max(cv_score_list))
    trained_clf = trained_clf_list[select_clf_index]
    
    # Test SVM
    test_predict = trained_clf.predict(test_images_flat)

    test_precision = precision_score(test_labels, test_predict, average='macro')
    test_recall = recall_score(test_labels, test_predict, average='macro')
    test_f1_score = f1_score(test_labels, test_predict, average='macro')
    test_accuracy = accuracy_score(test_labels, test_predict)

    file_name = rbf_folder_name + 'C_' + str(c) + '_no_PCA' + '.txt'
    print("Printing cross validation results to file: " + file_name)
    print_train_stats(file_name, output)
    print("Printing test results to file: " + file_name)
    print_test_stats(file_name, test_labels, test_predict, test_precision, test_recall, test_f1_score, test_accuracy)
    print('')

    # SVM on data which has different level of dimension reduction from PCA
    for pca_para in pca_para_list:
        print("Starting trainging with PCA of n_components: ", int(pca_para))
        pca = PCA(n_components=int(pca_para))

        x_train_trans = pca.fit_transform(train_images_flat[:50000])
        x_test_trans = pca.transform(test_images_flat[:10000])
        print('Information kept: ', sum(pca.explained_variance_ratio_)*100, '%')
        print('Noise variance: ', pca.noise_variance_)

        # Train SVM
        clf = svm.SVC(kernel='rbf', C=c)
        output = cross_validate(clf, x_train_trans, train_labels, cv=5, scoring=scoring, return_estimator=True)
        
        cv_score_list = output['test_precision_macro'].tolist()
        trained_clf_list = output['estimator']
        select_clf_index = cv_score_list.index(max(cv_score_list))
        trained_clf = trained_clf_list[select_clf_index]
        
        # Test SVM
        test_predict = trained_clf.predict(x_test_trans)

        test_precision = precision_score(test_labels, test_predict, average='macro')
        test_recall = recall_score(test_labels, test_predict, average='macro')
        test_f1_score = f1_score(test_labels, test_predict, average='macro')
        test_accuracy = accuracy_score(test_labels, test_predict)

        file_name = rbf_folder_name + 'C_' + str(c) + '_PCA_' + pca_para + '.txt'
        print("Printing cross validation results to file: " + file_name)
        print_train_stats(file_name, output)
        print("Printing test results to file: " + file_name)
        print_test_stats(file_name, test_labels, test_predict, test_precision, test_recall, test_f1_score, test_accuracy)
        print('')
        
print('')