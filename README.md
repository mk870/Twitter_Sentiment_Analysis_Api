# Twitter Sentiment Analysis AI
<img src="https://i.ibb.co/gMPZN0V/twitter.jpg" alt="twitter" border="0">  

## Project Summary 
* This application takes in a search input then processes it to return a live twitter sentiment score on that search word and also returns atleast 80 wordCloud topwords (most common words that people are saying based on the search word).  
* Used the sentiment140 dataset from kaggle (https://www.kaggle.com/kazanova/sentiment140) as my training data.
* Ended up using 50 000 data points, 25 000 for each target variable (positive and negative sentiments). This was done due to limited computational resources from google colab.
* Ran the entire data analysis, data cleaning and model building on google colab platform for added computational power.
* Engineered and cleaned the data so as to extract clean and clear tweets.
* Used Nltk library (NLP) to tokenize and stem the text data.
* Built a client facing Api using flask and hosted it on Heroku  
## The following is the Api in action:

<img src="https://i.ibb.co/zZb4zsf/Movie-Plus.png" alt="Movie-Plus" border="0">  

### **Resources Used**
***
**Python Version**: 3.8

**Packages**: Pandas, Numpy, Sklearn, Json, Flask, Pickle, Twitter API ,Kaggle, Jupyter notebook ,NLTK package and WordCloud.  
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white) ![Twitter](https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=flat&logo=Twitter&logoColor=white) ![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=flat&logo=googledrive&logoColor=white)

**For Web Framework Requirements**: pip install -r requirements.txt

**APIs**: Twitter API

### **Data Collection**
***
Used Kaggle Sentiment140 dataset with 1.6million tweets with 3 target variables positive,negative and neutral (https://www.kaggle.com/kazanova/sentiment140).  
Due to lack of computational resources, 50 000 tweets were used with only 2 target variables positive and negative with 25 000 tweets each.

### **Data Cleaning**
After data collection, data cleaning commenced with the following steps: 
* Removed Stop words using the NLTK library with the english language setting.
* Removed punctuations using the python String package.
* Removed repeating words, numbers and URLs using python regex.
* Tokenized the tweets using NLTK tokenizer. 
* Stemmed the tweets using NLTK PorterStemmer.
* Combined the words in each tweet for model building using the string method join.
* Seperated the dataframe into features (tweets) and targets (sentiments).


### **Model Building**
***
* Split the data into training and test sets at 70:30 ratio using the Sklearn library.
* Used the TFIDFVectoriser with 500 000 max-features and transformed  the training dataset.
* Used 2 models, the BernoulliNB model from naive bayes and the SupportVectorMachine model.
* Trained both models and tested using the test data leading to the following results:  
BernoulliNB model : **Precision** 73% , **Recall** 78%, **Accuracy** 74%.  
Support Vector Machine (SVM): **Precision** 75% , **Recall** 77%, **Accuracy** 75%.
* Picked the SVM model with 75% accuracy, vectorized the whole dataset and trained the model on the entire dataset.
* Saved the trained final model and the vectorized whole dataset.

### **Productionization**
***

In this step, I built a flask API endpoint thats hosted on Heroku. I did the following:
* Created a flask backend application.
* Created a Get endpoint which takes in a query from the client.
* The twitter api is used to get 160 tweets related to the client query.
* The same data cleaning functions which were used in the previous data cleaning steps are now utilized to clean the new incoming tweets.
* The saved vectorizer is used to transform the incoming tweets and feed them to the saved SVM model for sentiments.

* WordCloud library is used to acquire topwords (80 max) on the tweets before the data is vectorized.
* The endpoint returns the percentage of positive sentiments from the 160 new tweets and the topwords.

**Live Implemantation:** [MoviePlus](https://react-movieplus.netlify.app)
