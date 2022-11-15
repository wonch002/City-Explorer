from tabpy.tabpy_tools.client import Client

client = Client(r'https://localhost:9004/')

import tabpy_client
client=tabpy_client.Client('localhost:9004')

def add(x, y):
    import numpy as np
    return [np.add(x, y)]


client.deploy(name='add',
              obj=add,
              description='Adds two numbers x and y',
              schema={(1, 2): 3},
              override=True)

client.get_endpoints()
