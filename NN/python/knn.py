from util import *
from run_knn import run_knn

def get_perc(test, target):
    total = 0
    correct = 0
    for i in range(len(test)):
        if test[i][0] == target[i][0]:
            correct += 1
        total += 1
    return correct * 1.0/(total * 1.0) * 100


def main():
    train_data, valid_data, test_data, train_labels, valid_labels, test_labels = LoadData('digits.npz')
    print train_data.shape
    print train_labels.shape

    print valid_data.shape
    print valid_labels.shape

    print test_data.shape
    print test_labels.shape
    
    print "VALIDATION"
    for i in [1, 3, 5, 7, 9]:
        labels = run_knn(i, train_data.T, train_labels, valid_data.T)
        #plot_digits(test)
        print "K = ", i, " perc = ", get_perc(labels, valid_labels.T)

    print "TEST"
    for i in [1, 3, 5, 7, 9]:
        labels = run_knn(i, train_data.T, train_labels, test_data.T)
        #plot_digits(test)
        print "K = ", i, " perc = ", get_perc(labels, test_labels.T)


if __name__ == '__main__':
    main()


