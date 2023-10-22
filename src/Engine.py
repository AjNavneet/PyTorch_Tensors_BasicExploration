#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import torch
import pandas as pd
from MLPipeline.Arithmetic import Arithmetic
from MLPipeline.Gradient import Gradient

# # PyTorch: Tensors

# ## Contents:
# 1. Create Tensors
# 2. Tensor Attributes
# 3. Arithmetic Operations
# 4. Trigonometric Operations
# 5. Functions Using Tensors
# 6. Gradients

# ### 1. Create Tensors

# #### 1-D Tensor

# Create a one-dimensional tensor
x_list = [1, 2, 3]
x = torch.tensor(x_list)
print(x)

# Tensor Attributes

# Shape
print('Tensor shape:', x.shape)

# Data Type
print('Tensor data type:', x.dtype)

# Device for Operations
print('Tensor operations on device:', x.device)

# #### 2-D Tensor (Matrix)

x_list = [[1, 2, 3], [4, 5, 6]]
x = torch.tensor(x_list)

# Tensor Attributes

# Shape
print('Tensor shape:', x.shape)

# Data Type
print('Tensor data type:', x.dtype)

# Device for Operations
print('Tensor operations on device:', x.device)

# #### 3-D Tensor

x_list = [[[1], [2]], [[3], [4]], [[5], [6]]]
x = torch.tensor(x_list)

# Tensor Attributes

# Shape
print('Tensor shape:', x.shape)

# Data Type
print('Tensor data type:', x.dtype)

# Device for Operations
print('Tensor operations on device:', x.device)

# #### Empty Tensor

# Create an empty tensor
x = torch.ones((6))
print('Empty tensor:\n', x)

# Modify the tensor through indexing and assignment
for i in range(6):
    x[i] = x[i] + 3

print('Modified tensor:\n', x)

# ## Creating Tensors from Dataset

# Convert features of a dataset into a tensor
data_path = "../Input"

# Read the dataset stored as CSV using pandas
file_path = data_path + '/churn_data.csv'

# Read CSV file
df = pd.read_csv(file_path)

# Features of the dataset
columns_names = df.columns.values
print('The dataset has {} attributes:\n{}'.format(len(columns_names), columns_names))

# Create a tensor using the 'weekly_mins_watched' feature in the dataset
x = torch.tensor(df[['weekly_mins_watched', 'minimum_daily_mins', "maximum_daily_mins"]].values)

# Tensor Shape
print(x.shape)

# Data Type
print(x.dtype)

# ## Arithmetic Operations

# Perform arithmetic operations on the tensor
Arithmetic().operations(x)

# ## Gradient

# Compute gradients
Gradient().compute(df)
