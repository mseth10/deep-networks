
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

def Fij(X,W): # Sigmoid function
    sigmoid = 1 / (1 + np.exp(-X.dot(W.T)))
    return np.append(sigmoid,np.ones((len(sigmoid),1)),1)


# In[7]:

def Fij_prime(X,W): # Sigmoid derivative
    sigmoid = Fij(X,W)
    return np.multiply(sigmoid,1-sigmoid)


# In[8]:

def Fjk(Z,Wjk): # Softmax function
    num = np.exp(Z.dot(Wjk.T)).T
    den = num.sum(axis=0)
    return np.divide(num,den).T


# In[9]:

def accuracy(X,T,Wij,Wjk):
    Z = Fij(X,Wij)
    Y = Fjk(Z,Wjk)
    pred = np.mat(np.argmax(Y,axis=1)).T
    lbls = np.mat(np.argmax(T,axis=1)).T
    return float(sum(lbls == pred))/len(lbls)*100


# In[187]:

J = 21 # Number of hidden features including bias
X_valid = X[50000:]
T_valid = T[50000:]
Wij = np.random.randint(-1,high=2, size=(J-1,785))*0.02
Wjk = np.random.randint(-1,high=2, size=(10,J))*0.02
idx = random.sample(range(0,50000),1)
X_train = X[idx,:]
T_train = T[idx,:]
K = np.argmax(T_train,axis=1)
sum_EntropyDiff1 = sum_EntropyDiff2 = 0
for i in range(785):
    for j in range(J-1):
        Wij_plus = np.copy(Wij)
        Wij_minus = np.copy(Wij)
        Wij_plus[j][i] = Wij_plus[j][i] + 0.01
        Wij_minus[j][i] = Wij_minus[j][i] - 0.01
        Z_plus = Fij(X_train,Wij_plus)
        Y_plus = Fjk(Z_plus,Wjk)
        Z_minus = Fij(X_train,Wij_minus)
        Y_minus = Fjk(Z_minus,Wjk)
        EntropyDiff = np.log(Y_plus[0][K])-np.log(Y_minus[0][K])
        sum_EntropyDiff1 = sum_EntropyDiff1 + abs(EntropyDiff)
Z = Fij(X_train,Wij)
for j in range(J):
    for k in range(10):
        Wjk_plus = np.copy(Wjk)
        Wjk_minus = np.copy(Wjk)
        Wjk_plus[k][j] = Wjk_plus[k][j] + 0.01
        Wjk_minus[k][j] = Wjk_minus[k][j] - 0.01
        Y_plus = Fjk(Z,Wjk_plus)
        Y_minus = Fjk(Z,Wjk_minus)
        EntropyDiff = np.log(Y_plus[0][K])-np.log(Y_minus[0][K])
        sum_EntropyDiff2 = sum_EntropyDiff2 + abs(EntropyDiff)
avg_EntropyDiff = (sum_EntropyDiff1+sum_EntropyDiff2)/(J*10+785*(J-1))
print avg_EntropyDiff/0.02


# In[ ]:



