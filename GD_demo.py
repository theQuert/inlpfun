import numpy as np
import matplotlib.pyplot as plt

x_data = [ 338., 333., 328., 207., 226., 25., 179., 60., 208.,  606. ]
y_data = [ 640., 633., 619., 393., 428., 27., 193., 66., 226., 1591. ]
# y_data = w * x_data + bias 

x = np.arange(-200, -100, 1) # bias
y = np.arange(-5, 5, 0.1) # weight
Z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)

for i in range(len(x)):
    for j in range((len(y))):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
           Z[j][i] = Z[j][i] +(y_data[n] - b - w*x_data[n])**2
        Z[j][i] = Z[j][i]/len(x_data)
        
b = -129 # intialize b
w = -4 # intialize w
lr = 0.0000001 # learning rate
iteration = 100000

# Store intial values for plotting
b_history = [b]
w_history = [w]

# Iteration
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0*(y_data[n] - b - w*x_data[n])*1.0
        w_grad = w_grad - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
    
    # Update parameters
    b = b - lr * b_grad
    w = w - lr * w_grad
    
    # Store the parameters for plotting
    b_history.append(b)
    w_history.append(w)

# plot the figure
plt.contour(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()

## Change Learning_Rate to Learning_Rate*10
b = -129 # intialize b
w = -4 # intialize w
lr = 0.000001 # learning rate
iteration = 100000

# Store intial values for plotting
b_history = [b]
w_history = [w]

# Iteration
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0*(y_data[n] - b - w*x_data[n])*1.0
        w_grad = w_grad - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
    
    # Update parameters
    b = b - lr * b_grad
    w = w - lr * w_grad
    
    # Store the parameters for plotting
    b_history.append(b)
    w_history.append(w)

# plot the figure
plt.contour(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()

## Change Learning_Rate to Learning_Rate*10
b = -129 # intialize b
w = -4 # intialize w
lr = 0.00001 # learning rate
iteration = 100000

# Store intial values for plotting
b_history = [b]
w_history = [w]

# Iteration
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0*(y_data[n] - b - w*x_data[n])*1.0
        w_grad = w_grad - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
    
    # Update parameters
    b = b - lr * b_grad
    w = w - lr * w_grad
    
    # Store the parameters for plotting
    b_history.append(b)
    w_history.append(w)

# plot the figure
plt.contour(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()

    ## Using adagrad to solve the problem
'''
If increasing the iteration to speed up the rate of convergence, that would spped down actually.
Cause it have to calculate gradient at each iteration, but that would cost lots of time.
Therefore, we modify the learning rate, and change improve rate of convergence at meantime.
'''
b = -129
w = -4
lr = 1
iteration = 100000

b_lr = 0.0
w_lr = 0.0

# Store initial values for plotting
w_history = [w]
b_history = [b]

# Iteration
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0*(y_data[n] - b - w*x_data[n])*1.0
        w_grad = w_grad - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
    
    # Add
    b_lr = b_lr + b_grad ** 2
    w_lr = w_lr + w_grad ** 2
    
    # Update parameters
    b = b - lr /np.sqrt(b_lr) * b_grad
    w = w - lr /np.sqrt(w_lr) * w_grad
    
    # Store the parameters for plotting
    b_history.append(b)
    w_history.append(w)

# plot the figure
plt.contour(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()