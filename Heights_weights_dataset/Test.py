import matplotlib.pyplot as plt
import csv
from sklearn import linear_model
import numpy as np

def main():
	#Starting this to see how i stack up against 'normal', aka median weight of the people of 20 years ago (Assuming people were lighter back then)
	height = [] #inches
	weight = []	#pounds
	index  = []
	with open("SOCR Data MLB HeightsWeights.csv") as datafile:
		plots = csv.reader(datafile, delimiter = ';')
		for rows in plots:
			height.append(float(rows[1]))
			weight.append(float(rows[2]))
			index.append(rows[0])
	#Transform from imperial to metric 1"= 2.54cn
	for idx, (height_value, weight_value) in enumerate(zip(height, weight)):
		height[idx] = height[idx] * 2.54
		weight[idx] = weight[idx] / 2.205

	personal_weight = 75
	personal_height = 174
	#create a lin reg method to see reference weight
	regr = linear_model.LinearRegression()

	#data split for testing and training 20/80 split
	train_split = len(weight)// (1.25) #splitting the data to 80% training
	weight_train = weight[0:int(train_split)]
	weight_test  = weight[int(train_split):]
	weight_test  = map(float, weight_test)
	height_train = height[0:int(train_split)]
	height_test  = height[int(train_split):]
	#regression
	height_array = np.vstack((index[0:int(train_split)], height_train)).T
	weight_array = np.vstack((index[0:int(train_split)], weight_train)).T
	regr.fit(height_array ,weight_array)
	weight_test_array = np.vstack((map(float, (index[int(train_split):])), height_test)).T 
	#regr.fit(np.array(height_train).reshape(-1, 1), np.array(weight_train).reshape(-1, 1))
	#prediction
	weight_predictor = regr.predict(weight_test_array)


	#Plot them
	#plt.plot(height_test, weight_predictor, color='blue', linewidth=3)

	plt.scatter(height, weight)
	plt.scatter(personal_height, personal_weight, color = 'r')
	#plt.scatter(height_test, weight_predictor, color = 'b')
	plt.show()


if __name__ == '__main__':
	main()