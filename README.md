# Final Project:

## Runescape Grand Exchange Analysis & Supervised Machine Learning Bot
The Grand Exchange, commonly referred to as the GE, is a trading system for players to purchase and sell tradeable items in Old School RuneScape.

### Folder 1: Scrapper, Analyisis, ML
First we test our connection to the OSRS Grand Exchange API. in Test.py
Once we confirm our connection, we build a scrapper.py to connect to the OSRS grand exchange API, to export out the runes into a csv file


We Analyse over csv file with 1 year of grand exchange data sourced via the runescape Grand Exchange API. We do exploratory data analysis with Data_exploration.ipynb

We will be using python to help us looking at historical price's from 10 previous days, in order to try and predict the next day's price. We first need normalize our values. We do this by using the coefficent of variation formula. We split our data inti training and testing sets in order to apply a simple lstm model. We use Tensorflow to train our model to look 10 days into the past, in order to predict 1 day into the future the price of runes. ge_ml.ipynb

### Folder 2: Flask App





### Folder 3. Tableau & ETL


Tableau workbook
Embed .HTML 

#### With PgAdmin Store our Data in AWS RDS
![Screenshot 2021-11-04 205625](https://user-images.githubusercontent.com/83923903/140456262-7cb40cc1-a4b7-4b65-933b-e5e5146eeb09.png)

#### We Use SQL to create our Table 
![Screenshot 2021-11-04 210614](https://user-images.githubusercontent.com/83923903/140456662-a3ab6c3d-43bc-4c4d-ad65-ebdfd8137176.png)

#### Create our POSTGRESSQL within Tableau and create Dashboard
![Screenshot 2021-11-04 205625](https://user-images.githubusercontent.com/83923903/140456711-fcd2e8c8-a0c1-4201-9eae-93fe90f398ee.png)

![Screenshot 2021-11-04 205642](https://user-images.githubusercontent.com/83923903/140456753-906a7749-f01c-481f-a971-311ac9132c33.png)



## Skills

* Python (Pandas, matplotlib, tensorflow, seaborn)
* SQL
* API
* Tableau
* Extract, Transform Load
* Machine Learning
* S3



#### Anaconda Enviornment Set up

* Install pandas
* Install requests
* Install matplotlib
* Install seaborn
* Install Tensorflow



