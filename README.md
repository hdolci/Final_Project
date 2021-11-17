# Final Project:

## Runescape Grand Exchange Analysis & Supervised Machine Learning Bot
The Grand Exchange, commonly referred to as the GE, is a trading system for players to purchase and sell tradeable items in Old School RuneScape.

### Folder 1: Scrapper, Analyisis, ML
First we test our connection to the OSRS Grand Exchange API. in Test.py
Once we confirm our connection, we build a scrapper.py to connect to the OSRS grand exchange API, to export out the runes into a csv file


We Analyse over csv file with 1 year of grand exchange data sourced via the runescape Grand Exchange API. We do exploratory data analysis with Data_exploration.ipynb

We will be using python to help us looking at historical price's from 10 previous days, in order to try and predict the next day's price. We first need normalize our values. We do this by using the coefficent of variation formula. We add other runes as features to train our model. Next, We split our data inti training and testing sets in order to apply a simple vanilla lstm model. We create a Simple LMTS model a single hidden layer of LSTM units, and an output layer used to make a prediction. We do this with Tensorflow. To train our model we will look 10 days into the past, in order to predict 1 day into the future the price of runes. ge_ml.ipynb

### Folder 2: Flask App, HTML

Webpage for user to input a rune type, and output ML prediction into a Webpage. As well as Tableau Dashboard

#### User Input:
![Screen Shot 2021-11-16 at 8 33 47 PM](https://user-images.githubusercontent.com/83923903/142135990-380bfc96-7add-48fc-a6fa-7198f20f8b1f.png)

#### ML Output:
![Screen Shot 2021-11-16 at 8 33 59 PM](https://user-images.githubusercontent.com/83923903/142136057-29e2ed95-7d97-4eee-854b-349aedb9c24c.png)




### Folder 3. AWS PostgreSQL, Tableau, and ETL


Tableau workbook
Embed .HTML 

#### Use AWS to Create our RDS
<img width="652" alt="Screen Shot 2021-11-16 at 9 00 56 AM" src="https://user-images.githubusercontent.com/83923903/142031152-03d5042a-8e30-4a68-8ecd-51ed0c8a0c23.png">

#### Using PGadmin4 to connect to our RDS server
<img width="814" alt="Screen Shot 2021-11-16 at 8 58 31 AM" src="https://user-images.githubusercontent.com/83923903/142030478-861eeb2d-4884-45a6-8497-25e879faee00.png">


#### We Use SQL to create our Table and import Rune_data_ETL.csv 
![Screenshot 2021-11-04 210614](https://user-images.githubusercontent.com/83923903/140456662-a3ab6c3d-43bc-4c4d-ad65-ebdfd8137176.png)

#### We connect our POSTGRESSQL with Tableau and create our Dashboard
![Screenshot 2021-11-04 205625](https://user-images.githubusercontent.com/83923903/140456711-fcd2e8c8-a0c1-4201-9eae-93fe90f398ee.png)

![Screenshot 2021-11-04 205642](https://user-images.githubusercontent.com/83923903/140456753-906a7749-f01c-481f-a971-311ac9132c33.png)



## Skills

* Python (Pandas, matplotlib, tensorflow, seaborn)
* SQL
* API
* Tableau
* Extract, Transform Load
* Supervised Machine Learning
* AWS



#### Anaconda Enviornment Set up

* Install pandas
* Install requests
* Install matplotlib
* Install seaborn
* Install Tensorflow



