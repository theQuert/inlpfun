import numpy as np

# Original function of Softmax, but float error occurs
# def softmax(x):
#     exp_x = np.exp(x)
#     softmax_x = exp_x / np.sum(exp_x)
#     return softmax_x 

# softmax_ans = softmax([1000, 2000, 3000])
# print(softmax_ans)

# Fix float error for Softmax: softmax(x) = softmax(x+c)
# def softmax(x):
# 	x = x - np.max(x)
# 	exp_x = np.exp(x)
# 	softmax_x = exp_x / np.sum(exp_x)
# 	return softmax_x

# softmax_test = softmax([1000, 2000, 3000])
# print(softmax_test)

def softmax(double x):
	x = [1000, 2000, 3000]
	orig_shape = x.shape
	if len(x.shape) > 1:
		# Matrix
		# exp_minmax = lambda x: np.exp(x - np.max(x))
		denom = lambda x: 1.0 / np.sum(x)
		# x = np.apply_along_axis(exp_minmax, 1, x)
		denominator = np.apply_along_axis(demon, 1, x)

		if len(denominator.shape) == 1:
			denominator = denominator.reshsape((denominator.shape[0], 1))

		x = x * denominator
	else:
		# Vector
		x_max = np.max(x)
		x = x - x_max
		numerator = np.exp(x)
		denominator = np.sum(numerator)
		x = numerator.dot(denominator)


	assert x.shape == orig_shape
	return x

