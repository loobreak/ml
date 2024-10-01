import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data = pd.read_csv('../data/world-happiness-report-2017.csv')
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

num_iterations = 500
alpha = 0.01

linear_regression = LinearRegression(x_train, y_train, polynomial_degree=4, sinusoid_degree=5)
(theta, loss) = linear_regression.train(alpha, num_iterations)
print('Theta: ', theta)
print('start Loss: ', loss[0])
print('end Loss: ', loss[-1])

predictions_num = 100
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num)
y_predictions = linear_regression.predict(x_predictions.reshape(predictions_num, 1))

plt.scatter(x_train, y_train, color='blue', label = 'train data')
plt.scatter(x_test, y_test, color='red', label = 'test data')
plt.plot(x_predictions, y_predictions, color='green', label = 'predictions')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('World Happiness Report 2017')
plt.legend()
plt.show()

plt.plot(range(num_iterations), loss)
plt.xlabel('Iterations')
plt.title('Loss')
plt.show()



