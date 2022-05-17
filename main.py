# TODO implement the PINN3.0 Code
import torch
import torch.optim as optim
import torch.profiler
import torch.utils.data
from scripts.file_readers import file_reader

### naming conventions 
# Function - my_function
# Variable - my_variable
# Class - MyClass
# constant - MY_CONSTANT
# module - my_module.py
# package - mypackage

# If line breaking needs to occur around binary operators, like + and *, it should occur before the operator
# PEP 8 recommends that you always use 4 consecutive spaces to indicate indentation.
# Line up the closing brace with the first non-whitespace character of the previous line 
# Surround docstrings with three double quotes on either side, as in """This is a docstring"""
# - Write them for all public modules, functions, classes, and methods.
# - Put the """ that ends a multiline docstring on a line by itself
# Use .startswith() and .endswith() instead of slicing

file = './data/2DSTEN/2DSTEN_mesh.vtu'
file_bc = './data/2DSTEN/2DSTEN_bnc.vtk'

x, y, z, _  = file_reader(file, mesh=True)
xb, yb, zb, _ = file_reader(file_bc, mesh=False)


print('jobs done')







