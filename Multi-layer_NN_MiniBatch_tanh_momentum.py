
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

def Fij(X,W): # tanh
    tan_h = 1.7159 * np.tanh(2.0/3*X.dot(W.T))
    return np.append(tan_h,np.ones((len(tan_h),1)),1)


# In[7]:

def Fij_prime(X,W): # tanh derivative
    tan_h_prime = 1.7159 * 2.0/3 * (1-np.tanh(2.0/3*X.dot(W.T))**2)
    return np.append(tan_h_prime,np.ones((len(tan_h_prime),1)),1)


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


# In[22]:

for i in range(5):
    J = 21 # Number of hidden features including bias
    X_valid = X[50000:]
    T_valid = T[50000:]
    Wij = np.random.normal(loc=0.0, scale=1.0/28, size=(J-1,785))
    Wjk = np.random.normal(loc=0.0, scale=(J-1)**(-0.5), size=(10,J))
    Gij = np.zeros((J, 785))
    Gjk = np.zeros((10,J))
    Rij = np.ones(J-1) # learning rate
    Rjk = np.ones(10) # learning rate
    maxima = 0
    iterations = 0
    while(True):
        idx = random.sample(range(0,50000),256) # mini-batch size can also be changed
        X_train = X[idx,:]
        T_train = T[idx,:]
        Z_train = Fij(X_train,Wij)
        Y_train = Fjk(Z_train,Wjk)
        del_k = T_train-Y_train
        del_j = np.multiply(del_k.dot(Wjk),Fij_prime(X_train,Wij))
        prevGjk = Gjk
        prevGij = Gij
        Gjk = -del_k.T.dot(Z_train)/len(X_train) + prevGjk*0.5 # change 0.5 to 0 to remove the effect of momentum
        Gij = -del_j.T.dot(X_train)/len(X_train) + prevGij*0.5 # change 0.5 to 0 to remove the effect of momentum
        Wjk = Wjk - np.multiply(Gjk,Rjk[:,np.newaxis])
        Wij = Wij - np.multiply(Gij[:J-1],Rij[:,np.newaxis])
        validAccuracy = accuracy(X_valid,T_valid,Wij,Wjk)
        iterations = iterations + 1
        if validAccuracy >= maxima:
            maxima = validAccuracy
            Wij_final = Wij
            Wjk_final = Wjk
            flag = 0
        elif flag<20: # can be increased to get better results
            flag = flag + 1
        else:
            break
        for l in xrange(J-1):
            if prevGij[l].dot(Gij[l].T) < 0:
                Rij[l] = Rij[l]*0.95
            else:
                Rij[l] = Rij[l]+0.05
        for l in xrange(10):
            if prevGjk[l].dot(Gjk[l].T) < 0:
                Rjk[l] = Rjk[l]*0.95
            else:
                Rjk[l] = Rjk[l]+0.05
    testAccuracy = accuracy(X_test,T_test,Wij_final,Wjk_final)
    print "No. of iterations =", iterations
    print "Accuracy on validation dataset =", maxima
    print "Accuracy on validation dataset =", testAccuracy


# In[ ]:



