#!/usr/bin/env python
# coding: utf-8

# In[11]:


import deepxde as dde
import numpy as np
import tensorflow as tf


# In[12]:


# Second derivative of displacement w.r.t x (bending curvature) ??? Should'nt is be bending moment???
def ddy(x, y):
    return dde.grad.hessian(y, x)    #dde=> deep xde library


# Third derivative of displacement w.r.t x (related to shear force) (main: wxxx(x,t) → measures the shear force,)
def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)

# p = lambda x: x
L = 1
q = 9.80665
EI = 11.367

# Point load magnitude and location (NEW)
P = 2.892  # NEW
x0 = 0.9315  # location of point load from the fixed end (NEW)


# In[13]:


def p(x):
    """Distributed load plus concentrated load approximation"""  # UPDATED
    width = 0.001  # small region to approximate point load (NEW)
    delta = (tf.experimental.numpy.heaviside(x - x0, 1.0) -
             tf.experimental.numpy.heaviside(x - x0 - width, 1.0)) / width  # NEW
    return q + P * delta  # UPDATED


# In[14]:


EI_material = lambda x: EI


# In[15]:


# Define PDE residual for the Euler-Bernoulli static beam equation
def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    # includes concentrated load via p(x)  # UPDATED
    return EI_material(x)*dy_xxxx + p(x)    #similar to PDE of EB, or eqn4.5


# Function to identify the left (fixed) boundary x = 0
def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)  #isclose() function, which checks whether two values are numerically close — accounting for small floating-point errors.


# Function to identify the right (free) boundary x = L
def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], L)  

# Exact analytical solution for comparison (used as ground truth)  #not required
#"""Analytical deflection for constant q and point load"""
def func(x):
    distributed = -q * x**2 / (24 * EI) * (6 * L**2 - 4 * L * x + x**2)
    part1 = -P * x**2 / (6 * EI) * (3 * x0 - x)
    part2 = -P * x0**2 / (6 * EI) * (3 * x - x0)
    return np.where(x <= x0, distributed + part1, distributed + part2)


# In[16]:


# Define the 1D spatial geometry of the beam (domain from 0 to L) #replace with FEM data
geom = dde.geometry.Interval(0, L) 

# Boundary condition: w(0) = 0 (displacement at fixed end)
#On the left boundary (boundary_l), enforce the condition that displacement w(x)=0.
bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)   #B in eqn 4.6 #lambda =>For any input x, return 0. 
#Even though there's no explicit variable named w, the output of the network is interpreted as w(𝑥)

# Boundary condition: dw/dx = 0 at fixed end (no rotation)
bc2 = dde.NeumannBC(geom, lambda x: 0, boundary_l)   #A in eqn 4.6

# Boundary condition: d^2w/dx^2 = 0 at free end (no bending moment)
bc3 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)

# Boundary condition: d^3w/dx^3 = 0 at free end (no shear force)
bc4 = dde.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)


# In[17]:


# Prepare the PINN training dataset: PDE + boundary conditions + test data
data = dde.data.PDE(   #creates the dataset for solving PDE
    geom,                   #geometry of the domain, defined earlier with geom class
    pde,                    # defined in def pde(x, y) earlier
    [bc1, bc2, bc3, bc4],   #contributes to Ef (C in Eqn 4.6)
    num_domain=20,          #This sets how many random collocation points to generate to evaluate PDE
    num_boundary=2,         # randonly pick 2 points near x=0, and 2 points near x=L to enforce all BCs
    ##DeepXDE internally uses slightly jittered/randomized samples to helpapply constraints more robustly amd avoid exact-precision floating point issues
    solution=func,          #analytical soln to compare, defined above. Not for training
    num_test=100,      #DeepXDE generates 100 test points to compare predictions with the exact solution
)


# In[18]:


# Define the neural network architecture: input, hidden, output layers
layer_size = [1] + [30] * 3 + [1]

# Use tanh activation function for nonlinearity
activation = "tanh"

# Use Glorot uniform initializer to avoid vanishing gradients
initializer = "Glorot uniform"

# Create the feedforward neural network
net = dde.maps.FNN(layer_size, activation, initializer)

# Create a PINN model using data and neural network
model = dde.Model(data, net)   

# Compile model with optimizer and metric
model.compile("adam", lr=0.001, metrics=["l2 relative error"])


# In[19]:


# Train the model using Adam optimizer
losshistory, train_state = model.train(epochs=30000, display_every=1000)


# In[20]:


dde.saveplot(losshistory, train_state, issave=False, isplot=True)


# In[ ]:




