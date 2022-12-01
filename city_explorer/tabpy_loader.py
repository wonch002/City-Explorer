"""Tabpy functions for Tableau."""
import os
import socket

from typing import Callable

from typing import Callable, Dict, List
from flask import Flask, request


import platform
import time
from tabpy.tabpy_tools.client import Client


# Defining Example Add Function
def add(x, y):
    """Adds two numbers together using numpy.add()"""
    import numpy as np

    return np.add(x, y).tolist()


class SimilarCitiesClient(Client):
    def __init__(self, server: str = "http://localhost", port: int = 9004):
        """Initalize tabpy server."""
        self._start_tabpy_server()
        endpoint = f"{server}:{port}/"
        super().__init__(endpoint)

    @staticmethod
    def _start_tabpy_server():
        """Start the Tabpy server by opening a terminal and running `Tabpy`."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", 9004))
        # Only start if the server is not already running
        if result != 0:
            print("Starting TabPy server.")
            if platform.system() == "Windows":
                assert (
                    os.system("start cmd /c tabpy") == 0
                ), "Tabpy server failed to start!"
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

    def deploy(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        override: bool = True,
    ):
        """Deploy a function to our client.

        Overwriting default deploy to leverage python doc strings and names.
        """
        super().deploy(
            name=name or func.__name__,
            obj=func,
            description=description or func.__doc__,
            override=override,
        )


# Start our client
client = SimilarCitiesClient()


def similar_cities_tabpy(
    cities: List[int],
    city_id: int,
    occupation_title: str,
    population: float,
    population_denisty: float,
    age: float,
    sex: float,
    rental_prices: float,
    house_prices: float,
    affordability: float,
    political_party: float,
    winter_temperature: float,
    spring_temperature: float,
    summer_temperature: float,
    fall_temperature: float,
    precipitation: float,
    snowfall: float,
    education: float,
) -> Dict[str, float]:
    """TabPy for similar cities."""
    import requests
    import pandas as pd

    HOSTNAME = "localhost"
    FLASK_PORT = 5001

    response = requests.get(
        url=f"http://{HOSTNAME}:{FLASK_PORT}/predict_similar_cities/",
        params=dict(
            city_id=city_id,
            occupation_title=occupation_title,
            population=population,
            population_denisty=population_denisty,
            age=age,
            sex=sex,
            rental_prices=rental_prices,
            house_prices=house_prices,
            affordability=affordability,
            political_party=political_party,
            winter_temperature=winter_temperature,
            spring_temperature=spring_temperature,
            summer_temperature=summer_temperature,
            fall_temperature=fall_temperature,
            precipitation=precipitation,
            snowfall=snowfall,
            education=education,
        ),
    )
    # result has the following structure is in this form
    # {
    #   city_id (str): similarity_score (float),
    #   ...: ...
    # }

    result = response.json()

    # index = city_id, value = similarity_score
    filtered_cities = pd.Series(result).loc[cities]
    ordered_similarity_scores = filtered_cities.values.tolist()

    return ordered_similarity_scores


def start_tabpy():
    """Set up tabpy and deploy neccessary functions."""
    client = SimilarCitiesClient()

    # Deploy the neccessary functions
    client.deploy(func=similar_cities_tabpy)


# Deploy a flask application to do the heavy lifting.
app = Flask(__name__)


@app.route("/predict_similar_cities/", methods=["GET"])
def predict_similar_cities():
    """End point for predicting similar cities."""
    city_id = int(request.args.get("city_id"))
    occupation_title = str(request.args.get("occupation_title"))
    sliders = [
        float(request.args.get("population", default=1.0)),
        float(request.args.get("population_denisty", default=1.0)),
        float(request.args.get("age", default=1.0)),
        float(request.args.get("sex", default=1.0)),
        float(request.args.get("rental_prices", default=1.0)),
        float(request.args.get("house_prices", default=1.0)),
        float(request.args.get("affordability", default=1.0)),
        float(request.args.get("political_party", default=1.0)),
        float(request.args.get("winter_temperature", default=1.0)),
        float(request.args.get("spring_temperature", default=1.0)),
        float(request.args.get("summer_temperature", default=1.0)),
        float(request.args.get("fall_temperature", default=1.0)),
        float(request.args.get("precipitation", default=1.0)),
        float(request.args.get("snowfall", default=1.0)),
        float(request.args.get("education", default=1.0)),
    ]

    predictions = similar_cities.predict_similar_cities(
        city_id=city_id,
        occupation_title=occupation_title,
        sliders=sliders,
    )

    return predictions.to_json()


# Deploy the neccessary functions
client.deploy(func=add)
