# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import numpy.linalg as LA
import tensorflow as tf

from sklearn.model_selection import GridSearchCV
from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.classification.deep_learning.resnet import ResNetClassifier
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
from sktime.classification.deep_learning.tapnet import TapNetClassifier

import argparse
import csv

parser = argparse.ArgumentParser()

# Dataset Settings
parser.add_argument('--data_path', type=str, default="./Multivariate2018_ts/",
                    help='The path of data')
parser.add_argument('--dataset', type=str, default="BasicMotions",
                    help='MSTC dataset')

# Model Settings
parser.add_argument('--model', type=str, default="resnet",
                    help='Classifier of sktime. Default: resnet')
parser.add_argument('--use_curve', type=bool, default=True,
                    help='whether to use curve features. Default: True')

args = parser.parse_args()

# Curve computation
def matrix_inv(m):
    a1, b1, c1 = m[0, :, 0], m[0, :, 1], m[0, :, 2]
    a2, b2, c2 = m[1, :, 0], m[1, :, 1], m[1, :, 2]
    a3, b3, c3 = m[2, :, 0], m[2, :, 1], m[2, :, 2]
    determinant = a1*(b2*c3-c2*b3)-a2*(b1*c3-c1*b3)+a3*(b1*c2-c1*b2)
    adjoint_matrix = [[b2*c3-c2*b3, c1*b3-b1*c3, b1*c2-c1*b2],
                      [c2*a3-a2*c3, a1*c3-c1*a3, a2*c1-a1*c2],
                      [a2*b3-b2*a3, b1*a3-a1*b3, a1*b2-a2*b1]]
    determinant = np.array(determinant)
    determinant[np.where(determinant == 0)] = 0.000001
    inv = (1 / determinant) * np.array(adjoint_matrix)
    return inv

def PJCurvature_array_head3d_not_time3d(x, y, z):
    t_a = LA.norm((x[:, 1]-x[:, 0], y[:, 1]-y[:, 0], z[:, 1]-z[:, 0]), axis=0)
    t_b = LA.norm((x[:, 2] - x[:, 1], y[:, 2] - y[:, 1], z[:, 2] - z[:, 1]), axis=0)

    C = t_a.shape[0]
    m1 = np.concatenate((np.expand_dims(np.array([1]*C), 1), np.expand_dims(-t_a, 1), np.expand_dims(t_a ** 2, 1)), 1)
    m2 = np.concatenate((np.expand_dims(np.array([1] * C), 1), np.expand_dims(np.array([0] * C), 1), np.expand_dims(np.array([0] * C), 1)), 1)
    m3 = np.concatenate((np.expand_dims(np.array([1] * C), 1), np.expand_dims(t_b, 1), np.expand_dims(t_b ** 2, 1)), 1)
    M = np.concatenate((np.expand_dims(m1,0), np.expand_dims(m2,0), np.expand_dims(m3,0)), 0,)
    inv = matrix_inv(M)
    inv = inv.transpose(2, 0, 1)
    a = np.squeeze(np.matmul(inv, np.expand_dims(x, 2)))
    b = np.squeeze(np.matmul(inv, np.expand_dims(y, 2)))
    c = np.squeeze(np.matmul(inv, np.expand_dims(z, 2)))

    fenzi = np.sqrt((2 * b[:, 1] * c[:, 2]-2 * b[:, 2] * c[:, 1]) ** 2 + (2 * c[:, 1] * a[:, 2]-2 * c[:, 2] * a[:, 1]) ** 2 + (2 * a[:, 1] * b[:, 2]-2 * a[:, 2] * b[:, 1]) ** 2)
    fenmu = (a[:, 1] ** 2. + b[:, 1] ** 2. + c[:, 1] ** 2.) ** 1.5
    fenmu = np.where(fenmu == 0, 0.000001, 0) + fenmu
    kappa = fenzi / fenmu
    kappa = np.nan_to_num(kappa) #nan problem
    return kappa

def curve_curvature_array_head(yaw, pitch, roll):
    assert(len(yaw) == len(pitch))
    x = np.concatenate([np.expand_dims(np.array(yaw[0:-2]), 1), np.expand_dims(np.array(yaw[1:-1]), 1), np.expand_dims(np.array(yaw[2:]), 1)], 1)
    y = np.concatenate([np.expand_dims(np.array(pitch[0:-2]), 1), np.expand_dims(np.array(pitch[1:-1]), 1), np.expand_dims(np.array(pitch[2:]), 1)], 1)
    z = np.concatenate([np.expand_dims(np.array(roll[0:-2]), 1), np.expand_dims(np.array(roll[1:-1]), 1), np.expand_dims(np.array(roll[2:]), 1)], 1)
    kappa = PJCurvature_array_head3d_not_time3d(x, y, z)
    kappa = kappa.tolist()
    return kappa

def get_kappa_atten(fea):
    yaw, pitch, roll = fea[0,:], fea[1,:], fea[2,:]
    kappa = curve_curvature_array_head(yaw, pitch, roll)
    kappa = [0] + kappa + [0]
    return kappa

## dim_atten: The dataframe is converted to a tensor to compute and splice the curvature and then converted back to a dataframe.
def dim_atten(out_data, use_kappa=True, flatten_arctan=True):
    #Dataframe to tensor.
    #Get the shape of the dataframe.
    n, d = out_data.shape
    #Create an all-zero tensor of shape (n, length, d).
    length = out_data.applymap(lambda x: len(x)).max().max()
    tensor_shape = (n, length, d)
    tensor = np.zeros(tensor_shape)
    #Iterates over each element of the dataframe and assigns it to the corresponding tensor position.
    for i in range(n):
        for j in range(d):
            series_data = out_data.iloc[i, j]
            tensor[i, :len(series_data), j] = series_data

    #Convert NumPy arrays to tensorflow tensors.
    tensor = tf.convert_to_tensor(tensor)  # n*l*d
    tensor = tf.transpose(tensor, perm=[0, 2, 1])  # n*d*l
    dim_size = tf.shape(tensor)[1].numpy()
    add_dim = dim_size // 3

    #Calculate the curvature of the tensor and splice it.
    features = []
    for i, data in enumerate(tensor):
        data = tf.random.shuffle(data) #Dimensional randomisation
        tensor_kappa = np.array(data)
        for j in range(add_dim):
            if use_kappa:
                kappa = get_kappa_atten(data[3 * j:])
                if flatten_arctan:
                    kappa = np.arctan(kappa)
                tensor_kappa = np.concatenate((tensor_kappa, np.expand_dims(kappa, 0)), 0)
        features.append(tensor_kappa)
    features = np.array(features) #n*d_atten*l

    #Convert tensor to dataframe, the data format that matches the model inputs.
    #Convert tensor to numpy array
    array = np.array(features) # n*d_atten*l
    #Creates an empty dataframe.
    df = pd.DataFrame(index=range(n), columns=range(d +add_dim))

    # Iterate over each n*d_atten*l element.
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            #Converts series data of length l into DataFrame elements.
            series = pd.Series(array[i, j, :])
            # print("series: ", series)
            df.iloc[i, j] = series
            new_column_name = f'dims_{j}'
            df = df.rename(columns={j: new_column_name})
    df = df.applymap(lambda x: pd.Series(x))
    return df

path_train = args.data_path + args.dataset + "/" + args.dataset + '_TRAIN.ts'
path_test = args.data_path + args.dataset + "/" + args.dataset + '_TEST.ts'

# print("train_data", path_train)
# print("test_data", path_test)

X_train, y_train = load_from_tsfile_to_dataframe(
    full_file_path_and_name=path_train,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN"
)
X_test, y_test = load_from_tsfile_to_dataframe(
    full_file_path_and_name=path_test,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN"
)

if args.use_curve:
    X_train = dim_atten(X_train)
    X_test = dim_atten(X_test)

# The class labels
np.unique(y_train)

# Train a deep neural network classifier
if args.model == "resnet":
    network = ResNetClassifier(verbose=True)
    network.fit(X_train, y_train)
if args.model == "inception":
    network = InceptionTimeClassifier(verbose=True)
    network.fit(X_train, y_train)
if args.model == "tapnet":
    network = TapNetClassifier(verbose=True)
    network.fit(X_train, y_train)

acc = network.score(X_test, y_test)
with open('results.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([args.dataset, args.model, '{:.10f}'.format(acc)])
