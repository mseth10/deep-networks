from mnist import MNIST
import numpy as np
import csv
import matplotlib.pyplot as plt

def sigmoid(wx):
	return 1 / (1 + np.exp(-wx))

def error(imgs,lbls,w):
	return lbls.T - sigmoid(imgs.dot(w))

def loss(imgs,lbls,w):
	y = sigmoid(imgs.dot(w))
	return -(lbls).dot(np.log(y))-((1-lbls)).dot(np.log(1-y))

mndata = MNIST('./data')

img1,lbl1 = mndata.load_training()
img2,lbl2 = mndata.load_testing()

images1 = []
labels1 = []
images2 = []
labels2 = []

for i in range(10000):
	if lbl1[i] == 2 or lbl1[i] == 3:
		images1.append([1] + img1[i])
		labels1.append(1 - lbl1[i]/3)

for i in range(2000):
	if lbl2[i] == 2 or lbl2[i] == 3:
		images2.append([1] + img2[i])
		labels2.append(1 - lbl2[i]/3)

trainImgs = np.matrix(images1[0:1821])
trainLbls = np.matrix(labels1[0:1821])
validImgs = np.matrix(images1[1822:])
validLbls = np.matrix(labels1[1822:])
testImgs = np.matrix(images2)
testLbls = np.matrix(labels2)

lam = [0, 0.0001, 0.001, 0.01, 0.1]

y = []
z = []

for l in lam:
	w = np.random.randn(785, 1)*0.00001
	n0 = 0.000000001
	T = 10.0
	t = 0
	minima1 = len(validImgs)
	minima2 = len(testImgs)
	flag = 0
	x = 0

	while(True):
		
		trainError = error(trainImgs,trainLbls,w)
		#print np.amax(trainError), np.amin(trainError)
		validError = error(validImgs,validLbls,w)
		testError = error(testImgs,testLbls,w)

		trainIncorrect = float(sum(abs(trainError)>0.5)*100.0/len(trainImgs))
		validIncorrect = float(sum(abs(validError)>0.5)*100.0/len(validImgs))
		testIncorrect = float(sum(abs(testError)>0.5)*100.0/len(testImgs))
		#print trainIncorrect, validIncorrect, testIncorrect
		
		trainLoss = float(loss(trainImgs,trainLbls,w)/len(trainImgs))
		validLoss = float(loss(validImgs,validLbls,w)/len(validImgs))
		testLoss = float(loss(testImgs,testLbls,w)/len(testImgs))
		#print trainLoss,validLoss,testLoss

		data = [x,l,100-trainIncorrect,100-validIncorrect,100-testIncorrect]
		y.append(data)

		if validIncorrect <= minima1:
			minima1 = validIncorrect
			minima2 = testIncorrect
			w_final = w[1:]
			flag = 0
		elif flag<2:
			flag = flag + 1
		else:
			break
		
		grad = (trainImgs.T).dot(trainError) - l*len(trainImgs)*w/abs(w)
		#print np.amax(grad), np.amin(grad)
		n = n0/(1+t/T)
		t = t+1
		w = w+n*grad

		x = x+1

	data1 = [l, float(w_final.T.dot(w_final))]
	z.append(data1)


	
with open('plotdata4.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerows(y)

with open('plotdata6.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerows(z)
