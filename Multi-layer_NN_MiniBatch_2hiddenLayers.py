
# coding: utf-8

# In[1]:

import os
import struct
import numpy as np
import random


# In[2]:

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        print ('Done')
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)


# In[3]:

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


# In[4]:

data = list(read(dataset = "training", path = "./data"))
data2 = list(read(dataset = "testing", path = "./data"))


# In[5]:

N = len(data)
X = np.ones((N,785))
T = np.zeros((N,10))
for n in range(N):
    [lbl,img] = data[n]
    X[n][1:] = np.reshape((img - np.mean(img))/255,(784))
    T[n][lbl] = 1
N2 = len(data2)
X_test = np.ones((N2,785))
T_test = np.zeros((N2,10))
for n in range(N2):
    [lbl2,img2] = data[n]
    X_test[n][1:] = np.reshape((img2 - np.mean(img2))/255,(784))
    T_test[n][lbl2] = 1


# In[6]:

def Fj(X,W): # tanh
    tan_h = 1.7159 * np.tanh(2.0/3*X.dot(W.T))
    return np.append(tan_h,np.ones((len(tan_h),1)),1)


# In[7]:

def Fj_prime(X,W): # tanh derivative
    tan_h_prime = 1.7159 * 2.0/3 * (1-np.tanh(2.0/3*X.dot(W.T))**2)
    return np.append(tan_h_prime,np.ones((len(tan_h_prime),1)),1)


# In[8]:

def Fk(Z,W): # Softmax function
    num = np.exp(Z.dot(W.T)).T
    den = num.sum(axis=0)
    return np.divide(num,den).T


# In[9]:

def accuracy(X,T,Wij1,Wj1j2,Wj2k):
    Z1 = Fj(X,Wij1)
    Z2 = Fj(Z1,Wj1j2)
    Y = Fk(Z2,Wj2k)
    pred = np.mat(np.argmax(Y,axis=1)).T
    lbls = np.mat(np.argmax(T,axis=1)).T
    return float(sum(lbls == pred))/len(lbls)*100


# In[19]:

J = 21 # Number of hidden features
X_valid = X[50000:]
T_valid = T[50000:]
Wij1 = np.random.normal(loc=0.0, scale=1.0/28, size=(J-1,785))
Wj1j2 = np.random.normal(loc=0.0, scale=(J-1)**(-0.5), size=(J,J))
Wj2k = np.random.normal(loc=0.0, scale=(J-1)**(-0.5), size=(10,J))
Gij1 = np.zeros((J, 785))
Gj1j2 = np.zeros((J, J))
Gj2k = np.zeros((10,J))
Rij1 = np.ones(J-1) # learning rate
Rj1j2 = np.ones(J) # learning rate
Rj2k = np.ones(10) # learning rate
maxima = 0
iterations = 0
result = [['Iterations', 'Train Accuracy', 'Valid Accuracy', 'Test Accuracy']]
while(True):
    idx = random.sample(range(0,50000),256)
    X_train = X[idx,:]
    T_train = T[idx,:]
    Z1_train = Fj(X_train,Wij1)
    Z2_train = Fj(Z1_train,Wj1j2[:J-1])
    Y_train = Fk(Z2_train,Wj2k)
    del_k = T_train-Y_train #CEE
    del_j2 = np.multiply(del_k.dot(Wj2k),Fj_prime(Z1_train,Wj1j2[:J-1]))
    del_j1 = np.multiply(del_j2.dot(Wj1j2),Fj_prime(X_train,Wij1))
    prevGj2k = Gj2k
    prevGj1j2 = Gj1j2
    prevGij1 = Gij1
    Gj2k = -del_k.T.dot(Z2_train)/len(X_train) + prevGj2k*0.1
    Gj1j2 = -del_j2.T.dot(Z1_train)/len(X_train) + prevGj1j2*0.1
    Gij1 = -del_j1.T.dot(X_train)/len(X_train) + prevGij1*0.1
    Wj2k = Wj2k - np.multiply(Gj2k,Rj2k[:,np.newaxis])
    Wj1j2 = Wj1j2 - np.multiply(Gj1j2,Rj1j2[:,np.newaxis])
    Wij1 = Wij1 - np.multiply(Gij1[:J-1],Rij1[:,np.newaxis])
    validAccuracy = accuracy(X_valid,T_valid,Wij1,Wj1j2[:J-1],Wj2k)
    iterations = iterations + 1
    if validAccuracy >= maxima:
        maxima = validAccuracy
        Wij1_final = Wij1
        Wj1j2_final = Wj1j2[:J-1]
        Wj2k_final = Wj2k
        flag = 0
    elif flag<20:
        flag = flag + 1
    else:
        break
    for l in xrange(J-1):
        if prevGij1[l].dot(Gij1[l].T) < 0:
            Rij1[l] = Rij1[l]*0.95
        else:
            Rij1[l] = Rij1[l]+0.05
    for l in xrange(J):
        if prevGj1j2[l].dot(Gj1j2[l].T) < 0:
            Rj1j2[l] = Rj1j2[l]*0.95
        else:
            Rj1j2[l] = Rj1j2[l]+0.05
    for l in xrange(10):
        if prevGj2k[l].dot(Gj2k[l].T) < 0:
            Rj2k[l] = Rj2k[l]*0.95
        else:
            Rj2k[l] = Rj2k[l]+0.05
    trainAccuracy = accuracy(X[:50000],T[:50000],Wij1,Wj1j2[:J-1],Wj2k)
    testAccuracy = accuracy(X_test,T_test,Wij1,Wj1j2[:J-1],Wj2k)
    result.append([iterations, trainAccuracy, validAccuracy, testAccuracy])
testAccuracy = accuracy(X_test,T_test,Wij1_final,Wj1j2_final,Wj2k_final)
print "No. of iterations =", iterations
print "Accuracy on validation dataset =", maxima
print "Accuracy on test dataset =", testAccuracy


# In[33]:

import csv
with open('AccuracyPlot4layer.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(result)


# In[ ]:



