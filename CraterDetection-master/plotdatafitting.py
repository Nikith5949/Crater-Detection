"""
Flavio Andrade 5-11-18
This programs plots the test and validation accuracies.
"""
import numpy as np
import matplotlib as mpl

def plotDataFit(testData, validationData, EPOCHS, save, file_name="accuracies_graph"):
    if save == 1:
        mpl.use('Agg')

    import matplotlib.pyplot as plt

    mpl.style.use('seaborn')
    plt.title("Test vs Validation Accuracy", color='C0')
    plt.axis([0, EPOCHS, 0, 1])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    epochs = list(range(EPOCHS))
    line1, = plt.plot(epochs, testData)
    line2, = plt.plot(epochs, validationData)
    plt.legend([line1, line2], ['Test Data', 'Validation Data'])
    if save == 1:
        plt.savefig(file_name)
    else:
        plt.show()

def plotData(datax, datay, save, title, name):
    if save == 1:
        mpl.use('Agg')
    import matplotlib.pyplot as plt
    mpl.style.use('seaborn')
    plt.title(title + " Accuracy", color='C0')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    iters = list(range(datax))
    plt.plot(iters, datay)
    if save == 1:
        plt.savefig(name)
    else:
        plt.show()

if __name__ == "__main__":
    test = list(range(15))
    vals = list(range(15, 30))
    #test_numpy = np.array(test)
    #vals_numpy = np.array(vals)
    #plotDataFit([.98, .91, .95, .96, .99], [.93, .98, .91, .99, .93], 5, 1, "myfile")
    plotData(10, [1,2,3,4,5,6,7,8,9, 10], 0, "test", "testgraph")
