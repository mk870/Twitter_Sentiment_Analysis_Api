import pandas as pd
import pickle
import numpy as np
from flask import  Flask,request, jsonify, make_response
import re
import tweepy
import nltk
from flask_cors import CORS
import string
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')

with open('model.pkl','rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl','rb') as f:
    vectoriser = pickle.load(f)
api_key = 'ZmKvGdJ6a6VepKvq2fu35L8RR'
api_key_secret = '2NKQyMex4ga3fAeANdNOgjTugMg0SUWMGIKnRuJ8divF3O78hb'
access_token = '723255442479128576-oi3UEZq7HlWKMPrrP650QZE5C5dlflm'
access_token_secret = 'AkBBx73rYfXbT4ltIofFQb79LlUofBgEdwXpNetzobHyK'

auth_handler = tweepy.OAuthHandler(consumer_key=api_key, consumer_secret=api_key_secret)
auth_handler.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth_handler)
try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords.words('english')])

english_punctuations = string.punctuation
punctuations_list = english_punctuations

def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
tokenizer = RegexpTokenizer(r'\w+')
def stemming_on_text(data):
    st = nltk.PorterStemmer()
    text = [st.stem(word) for word in data]
    return text
    
def combine(data):
    return ' '.join(data)

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route("/music",methods = ['POST'])
def musicSentiment():
    
    if request.method == 'POST':
        try:
            req = request.get_json()
            search_term = req['search'] 
            tweetlist =[]
            tweets2 = tweepy.Cursor(api.search_tweets,q = search_term,lang = 'en').items(160)
            for tweet in tweets2:
                tweetlist.append(tweet.text)

            df = pd.DataFrame(tweetlist,columns =['Tweets'])
            df['text'] = df['Tweets'].apply(lambda text: cleaning_stopwords(text))
            df['text2']= df['text'].apply(lambda x: cleaning_punctuations(x))
            df['text3'] = df['text2'].apply(lambda x: cleaning_repeating_char(x))
            df['text4'] = df['text3'].apply(lambda x: cleaning_URLs(x))
            df['text5'] = df['text4'].apply(lambda x: cleaning_numbers(x))
            df['text6'] = df['text5'].apply(tokenizer.tokenize)
            df['text7']= df['text6'].apply(lambda x: stemming_on_text(x))
            df['text8'] = df['text7'].apply(lambda x: combine(x))
            processed_text = df['text8']
            
            final_data = vectoriser.transform(processed_text)
            predictions = model.predict(final_data)
            positive=[]
            negative = []
            for i in predictions:
                if i == 0:
                    negative.append(i)
                else:
                    positive.append(i)

            score = round((len(positive)/len(predictions))*100)
                    

            #wordcloud words
            commonwords = df['text5']
            wc = WordCloud(max_words = 80 , width = 1600 , height = 800,
                        collocations=False).generate(" ".join(commonwords))
            topwords = list(wc.words_.keys())

            res = make_response(jsonify({'sentiments':score,'wordcloud':topwords,'error':''}))
            return res
        except:
            res = make_response(jsonify({'sentiments':'','wordcloud':'','error': 'yes'}))
            return res

@app.route("/",methods = ['GET'])
def hello():
    return jsonify({"response":"hello this is a twitter sentiment app"})

if __name__== "__main__":
    app.run(debug = True)

