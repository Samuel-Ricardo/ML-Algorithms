


import numpy as np
import matplotlib.pyplot as plt


def generate_gradient(X, theta, y):
    sample_count = X.shape[0]
    return (1./sample_count)*X.T.dot(X.dot(theta) - y)


def get_training_data(file_path):
    orig_data = np.loadtxt(file_path, skiprows=1)
    cols = orig_data.shape[1]

    return  (orig_data, orig_data[:, :cols - 1], orig_data[:, cols-1:])


def init_theta(feature_count):
    return np.ones(feature_count).reshape(feature_count, 1)


def gradient_descending(X, y, theta, alpha):
    
    Jthetas = []
    Jtheta = (X.dot(theta)-y).T.dot(X.dot(theta) - y)

    index = 0

    gradient = generate_gradient(X, theta, y)

    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient

        gradient = generate_gradient(X, theta, y)

        Jtheta = (X.dot(theta) - y).T.dot(X.dot(theta) - y)

        if (index + 1) % 10 == 0:
            Jthetas.append((index, Jtheta[0]))

        index += 1
    return theta, Jthetas


def showJtheta(diff_value): 
    
    p_x = []
    p_y = []

    for (index, sum) in diff_value:
        p_x.append(index)
        p_y.append(sum)

    plt.plot(p_x, p_y, color='b')
    plt.xlabel('steps')
    plt.ylabel('loss_function')
    plt.title('Steps - loss function curve')

    plt.show()


def showlinecurve(theta, sample_training_set):

    x, y = sample_training_set[:, 1], sample_training_set[:, 2]
    z = theta[0] + theta[1] * x 

    plt.scatter(x,y, color='b', marker='x', label="Sample Data")
    plt.plot(x, z, 'r', color='r', label="Regression Cure")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title('linear regression curve')
    plt.legend()

    plt.show()


trainig_data_include_y, training_x, y = get_training_data("../docs/ML/ML/02/lr2_data.txt")
sample_count, feature_count = training_x.shape

alpha = 0.01

theta = init_theta(feature_count)
result_theta, Jthetas = gradient_descending(training_x, y, theta, alpha)

print("w:{}".format(result_theta[0][0]), "b:{}".format(result_theta[1][0]))

showJtheta(Jthetas)
showlinecurve(result_theta, trainig_data_include_y)


