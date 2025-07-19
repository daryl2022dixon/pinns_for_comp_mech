#!/usr/bin/env python
# coding: utf-8

# In[1]:


import deepxde as dde
import numpy as np
import tensorflow as tf
import pandas as pd


# In[2]:


# Second derivative of displacement w.r.t x (bending curvature)
def ddy(x, y):
    return dde.grad.hessian(y, x)  


# In[17]:


# Third derivative of displacement w.r.t x (related to shear force) 
def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)


# In[18]:


L = 0.8611 #cantilever length of beam in m
g = 3.177   #constant distributed load due to gravity per length N/m (depends on volume)
EI = 11.367  #Young's modulus x Moment of Inertia

P = 2.893  # point load magnitude in N
x0 = 0.7914  # location of point load from the fixed end


# In[19]:


# Load training coordinates from Excel file (first column)
train_df = pd.read_excel(
    "train_data_for PINN_disp_TS_v5.xlsx", skiprows=1, usecols=[0], header=None
)
anchors = train_df.values.astype(np.float32)


# In[20]:


anchors


# In[21]:


#function representing constant distributed plus concentrated load
def q(x):
    width = 0.01  # tweakable parameter
    delta = tf.exp(-((x - x0) ** 2) / (2 * width ** 2)) / (width * tf.sqrt(2 * np.pi))
    return g + P * delta


# In[22]:


EI_material = lambda x: EI


# In[9]:


#def pde(x, y):
    ## includes concentrated load via q(x)
    #return EI_material(x)*dy_xxxx + q(x)    #similar to PDE of EB, or eqn4.5


# In[23]:


# Define PDE residual for the Euler-Bernoulli static beam equation
def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    # includes concentrated load via q(x)
    return EI_material(x)*dy_xxxx + q(x)    #similar to PDE of EB, or eqn4.5


# In[24]:


# Function to identify the left (fixed) boundary x = 0
def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)  #isclose() function, which checks whether two values are numerically close — accounting for small floating-point errors.


# Function to identify the right (free) boundary x = L
def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], L) 


# In[25]:


# Exact analytical solution for comparison (used as ground truth)
#Analytical deflection for constant g and point load
def func(x):
    distributed = (-g * x**2 / (24 * EI)) * (6 * L**2 - 4 * L * x + x**2)
    part1 = (-P * x**2 / (6 * EI)) * (3 * x0 - x)
    part2 = (-P * x0**2 / (6 * EI)) * (3 * x - x0)
    return np.where(x <= x0, distributed + part1, distributed + part2)


# In[26]:


# Training points are provided via "anchors" loaded from the Excel file above
geom = dde.geometry.Interval(0, L)


# In[27]:


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


# In[28]:


anchors = np.array([[0.0], [L]])


# In[29]:


data = dde.data.PDE(   # creates the dataset for solving PDE
    geom,                   # geometry of the domain
    pde,                    # defined in def pde(x, y)
    [bc1, bc2, bc3, bc4],   # contributes to Ef (C in Eqn 4.6)
    num_domain=0,           # no randomly generated interior points
    num_boundary=2,         # randomly pick points near x=0 and x=L to enforce BCs
    anchors=anchors,        # additional fixed points supplied at the boundaries
    solution=func,          # analytical solution for comparison only
    num_test=100,           # DeepXDE generates test points for error evaluation
)


# In[30]:


# Define the neural network architecture: input, hidden, output layers
layer_size = [1] + [30] * 4 + [1]

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


# In[31]:


# Train the model using Adam optimizer
losshistory, train_state = model.train(epochs=30000, display_every=1000)


# In[32]:


dde.saveplot(losshistory, train_state, issave=False, isplot=True)


# In[ ]:




