import tensorflow as tf
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def one_hot(labels, C):
    """
    makes one hot from labels

    PARAMS
    ------
    labels: label array
    C: number of classes
    """
    C = tf.constant(C, name='C')
    one_hot_mat = tf.one_hot(labels, C, axis=0)
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_mat)

    return one_hot


def load_jpgs(fold, label_df, f,id_col, has_thing):
    """
    loads jpgs, labels into X_tr, Y_tr
    used for 1 class classification

    PARAMS
    ------------
    fold: train folder containing images
    label_df: dataframe with labels, and values
    id_col: Identificaion column ("ID" of image)
    has thing: 'has cactus', 'has __' etc
    """
    X_tr = []
    Y_tr = []
    imges = label_df[id_col].values
    for img_id in imges:
        X_tr.append(cv2.resize(cv2.imread(fold + img_id),(f,f)).astype(np.float32))
        Y_tr.append(label_df[label_df[id_col] == img_id][has_thing].values[0])

    X_tr = np.asarray(X_tr)
    X_tr = X_tr.astype('float32')
    X_tr /= 255
    Y_tr = np.asarray(Y_tr)

    return X_tr, Y_tr


def split(X_tr, Y_tr):
    """
    splits train,test data
    """
    X_train, y_train, X_test, y_test = train_test_split(X_tr, Y_tr)
    return X_train, y_train, X_test, y_test


def load_test_jpgs():
    """
    loads test files
    """
    X_te = []
    for i in os.listdir(test_fold):
        X_te.append(cv2.imread(test_fold + i))

    X_te = np.asarray(X_te)
    X_te = X_te.astype('float32')
    X_te /= 255
    return X_te
