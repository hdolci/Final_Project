# Final Project:

## Runescape Grand Exchange Analysis & Supervised Machine Learning Bot
The Grand Exchange, commonly referred to as the GE, is a trading system for players to purchase and sell tradeable items in Old School RuneScape.

### Folder 1. 
First we test our connection to the OSRS Grand Exchange API. in Test.py
Once we confirm our connection, we build a scrapper.py to connect to the OSRS grand exchange API, to export out the runes into a csv file


We Analyse over csv file with 1 year of grand exchange data sourced via the runescape Grand Exchange API. We do exploratory data analysis with Data_exploration.ipynb

We will be using python to help us looking at historical price's from 10 previous days, in order to try and predict the next day's price. We first need normalize our values. We do this by using the coefficent of variation formula. We split our data inti training and testing sets in order to apply a simple lstm model. We use Tensorflow to train our model to look 10 days into the past, in order to predict 1 day into the future the price of runes. ge_ml.ipynb

### Folder 2.
We will be scraping the Runescape World map in order to map out coordinates to instruct our ML model to move in the world state.



### Folder 3.

We will be using a runecrafting bot to create our selected rune.


### Folder 4. 
Link to Tableau Dashboard, Overall summary of our results for the past two weeks of botting. 


## Skills

* Python (Pandas, matplotlib, tensorflow, seaborn)
* SQL
* API
* Tableau
* ETL
* Supervised ML
* S3

### Resource
* Pycharm: https://www.jetbrains.com/pycharm/download/#section=mac

#### Anaconda Enviornment Set up

* Install pandas
* Install requests
* Install matplotlib
* Install seaborn
* Install Tensorflow


### Folder 1. Grand Exchange Analysis



### Folder 2. Webscrape & Mapping


### Folder 3. Supervised Machine Learning



### Folder 4. Tableau Dashboard & Flask App
* Logs_Data.csv
* stats.csv
* user.csv
