"""Tabpy functions for Tableau."""
import os
import socket

import platform
import time
from tabpy.tabpy_tools.client import Client


def start_tabpy_server():
    """Start the Tabpy server by opening a terminal and running `Tabpy`."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("localhost", 9004))
    # Only start if the server is not already running
    if result != 0:
        print("Starting TabPy server.")
        if platform.system() == "Windows":
            assert os.system("start cmd /c tabpy") == 0, "Tabpy server failed to start!"
        elif platform.system() == "Darwin":
            assert (
                os.system(
                    """
                    osascript -e 'tell app "Terminal" to do script "tabpy"'
                    """
                )
                == 0
            ), "Tabpy server failed to start!"
        else:
            raise NotImplementedError(f"{platform.system()} is not supported.")
        time.sleep(1)  # Give the server time to fully start
    else:
        print("TabPy server is already running.")

    print("TabPy server is running at http://localhost:9004/")


start_tabpy_server()

# Defining Client
client = Client("http://localhost:9004/")


# Defining Example Add Function
def add(x, y):
    """Adds two numbers together using numpy.add()"""
    import numpy as np

    return np.add(x, y).tolist()


# Deploy Example Add Function
client.deploy("add", add, "Adds together two inputs, x and y", override=True)
