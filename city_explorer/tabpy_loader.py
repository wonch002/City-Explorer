import os
from tabpy.tabpy_tools.client import Client

# Starting Tabpy Server - This should open a terminal with a tabpy instance
###----NEED TO VALIDATE FUNCTIONALITY ON MACOS----###
assert os.system("start cmd /c tabpy") == 0, "Tabpy server failed to start!"

# Defining Client
client = Client('http://localhost:9004/')


# Defining Example Add Function
def add(x, y):
    """Adds two numbers together using numpy.add()"""
    import numpy as np
    return np.add(x, y).tolist()


# Deploy Example Add Function
client.deploy('add', add, "Adds together two inputs, x and y", override=True)
