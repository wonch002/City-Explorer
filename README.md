# City Explorer

City Explorer is a visualization that provides a user-driven method for exploring 
cities in the United States. The general workflow is as follows:

1. Users define a city of interest (e.g., Atlanta, GA) and the characteristics of 
    that city that are meaningful to them.
2. City Explorer provides a set of recommended towns similar to the original city 
    of interest based on the initial characteristics.
3. Users adjust their preferences as they see fit to continue exploring similar
    cities.

The frontend visualization leverages Tableau to prompt user input and display
recommended cities (see `city_explorer/tableau/`). We created a Flask API to accept the
information from Tableau to compute a similarity score using a weighted euclidean
distance to deliver a set of recommended cities. Tableau and python are connected using
a TabPy server (see https://www.tableau.com/developer/tools/python-integration-tabpy).


# Installation

These instructions operate under the assumption that you have python and tableau
installed. If you don't, please install them from their respective websites.

1. Download the project locally

2. Launch a terminal
    Mac: `Command + Space Bar, Type Terminal and launch`
    Windows: `Launch search (the bottom left) and type "cmd". Launch the command prompt` 

3. Navigate to the project directory using the terminal (City-Explorer).

4. Install project dependencies by running `pip install -r requirements.txt`

5. Launch the TabPy and Flask server by running `python city_explorer/tabpy_loader.py`

6. Launch the Tableau workbook found at
   `City-Explorer/city_explorer_dashboard.twbx`

7. In the top navigation toolbar of the Tableau workbook, under Help -> Setting and 
Performance -> Manage Analytics Extention Connection select TabPy as connection type

8. Uncheck both "Require SSL" and "Sign in with username and password", set Hostname to
"localhost" and port to "9004", then select "Save".

# Demo

<TODO>


# Project Contributors
Georgia Institute of Technology

CSE6242 - Team 162

Ankit Anand, Matthew Tate, Krishna Tatineni, Cameron Wonchoba