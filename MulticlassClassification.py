from mnist import MNIST
import numpy as np
import numpy.matlib
import csv
import matplotlib.pyplot as plt

def weightGuess(wx):
	y = np.exp(wx)
	s = y.sum(axis=1)
	return y/np.matlib.repmat(s,1,10)

def loss(t,y,N):
	correct = 100.0*sum(np.argmax(t,axis=1)==np.argmax(y,axis=1))/N
	loss = -1.0*sum(np.multiply(t,np.log(y))).sum(axis=1)/N
	return float(correct), float(loss)

mndata = MNIST('./data')

img1,lbl1 = mndata.load_training()
img2,lbl2 = mndata.load_testing()

images1 = []
labels1 = np.zeros((10000,10))
images2 = []
labels2 = np.zeros((2000,10))

for i in range(10000):
	images1.append([1] + img1[i])
	labels1[i][lbl1[i]] = 1

for i in range(2000):
	images2.append([1] + img2[i])
	labels2[i][lbl2[i]] = 1

trainImgs = np.matrix(images1[0:8999])
trainLbls = np.matrix(labels1[0:8999])
validImgs = np.matrix(images1[9000:])
validLbls = np.matrix(labels1[9000:])
testImgs = np.matrix(images2)
testLbls = np.matrix(labels2)

lam = [0.001]

for l in lam:
	w = np.random.randn(785, 10)*0.0001
	n0 = 0.000000001
	T = 10.0
	t = 0
	minima1 = 0
	minima2 = 0
	flag = 0
	x = 0
	y = []

	while(True):
		
		trainGuess = weightGuess(trainImgs.dot(w))
		validGuess = weightGuess(validImgs.dot(w))
		testGuess = weightGuess(testImgs.dot(w))

		#print trainGuess[0]
		#print trainLbls[0]

		trainCorrect, trainLoss = loss(trainLbls,trainGuess,len(trainImgs))
		validCorrect, validLoss = loss(validLbls,validGuess,len(validImgs))
		testCorrect, testLoss = loss(testLbls,testGuess,len(testImgs))
		#print trainLoss,validLoss,testLoss

		data = [x,trainLoss,validLoss,testLoss,trainCorrect,validCorrect,testCorrect]
		y.append(data)

		if validCorrect >= minima1:
			minima1 = validCorrect
			minima2 = testCorrect
			flag = 0
		elif flag<2:
			flag = flag + 1
		else:
			break

		trainError = trainLbls - trainGuess
		#print np.amax(trainError), np.amin(trainError)
		grad = (trainImgs.T).dot(trainError) -  2*l*len(trainImgs)*w
		#print np.amax(grad), np.amin(grad)
		n = n0/(1+t/T)
		t = t+1
		w = w+n*grad

		x = x+1
		#print l, validCorrect, testCorrect
	print l, minima1, minima2

with open('softmax.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerows(y)
