
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import streamlit as st
import requests

# import libraries
# ======================================================
#import p7_nlp_preprocessing_local
import re
from string import punctuation
import contractions
from nltk import word_tokenize
from num2words import num2words
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from emoticons_local import UNICODE_EMO, EMOTICONS
# ======================================================
import pickle
from tensorflow import keras
from keras.utils import pad_sequences


# load the model
model_file_name = 'model_oneDNN_disabled.hdf5'
model = keras.models.load_model( model_file_name )

# load the tokenizer
tokenizer_file_name = 'saved_tokenizer_pickle.pkl'
tokenizer = pickle.load(open(tokenizer_file_name,'rb'))


# =========================================================================================== 
# Remove URL
def remove_urls(data):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', str(data))
    return data
# Remove USERNAME
def remove_username(data):
    username_pattern = re.compile(r'@\S+')
    data = username_pattern.sub(r'', str(data))
    return data
# Replaciong emojis with their corresponding sentiments
def emoji(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :') , :O
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|:O)', ' positiveemoji ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' positiveemoji ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' positiveemoji ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-; , @-)
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;|@-\))', ' positiveemoji ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:, :-/ , :-|
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:|:-/|:-\|)', ' negetiveemoji ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' negetiveemoji ', tweet)
    return tweet
# +++++++++++++++++++++++++++++++++++++++++++++
# Processing the tweet
def process_tweet_phase1(tweet):
    tweet = remove_username(tweet)                                    # Removes usernames
    tweet = remove_urls(tweet)                                        # Remove URLs
    tweet = emoji(tweet)                                               # Replaces Emojis
    return tweet
# +++++++++++++++++++++++++++++++++++++++++++++
from chat_words_local import chat_words_list, chat_words_map_dict
# Conversion of chat words
def convert_chat_words(data):
    tokens = word_tokenize(str(data))
    new_text = []
    for w in tokens:
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)
# EXPAND CONTRACTIONS
def expend_contractions(data):
    new_text = ""
    for word in str(data).split():
        # using contractions.fix to expand the shortened words
        #expanded_words.append(contractions.fix(word))  
    
        new_text = new_text + " " + contractions.fix(word)
    return new_text 
# REMOVE MAXIMUM    
def remove_maximum(data):
    data = re.sub(r'[^a-zA-z]', r' ', data)
    data = re.sub(r"\s+", " ", str(data))
    return data
# +++++++++++++++++++++++++++++++++++++++++++++
def process_tweet_phase2(tweet):    
    #tweet = convert_numbers(tweet)    
    tweet = convert_chat_words(tweet)
    tweet = expend_contractions(tweet)                                           
    tweet = tweet.lower()                                             # Lowercases the string
    tweet = re.sub(r"\d+", " ", str(tweet))                           # Removes all digits
    tweet = re.sub('"'," ", str(tweet))                               # Remove (") 
    tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))                   # Removes all single characters
    tweet = re.sub(r"[^\w\s]", " ", str(tweet))                       # Removes all punctuations
    tweet = re.sub(r'(.)\1+', r'\1\1', str(tweet))                    # Convert more than 2 letter repetitions to 2 letter
    tweet = re.sub(r"\s+", " ", str(tweet))                           # Replaces double spaces with single space    
    tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))                   # Removes all single characters
    tweet = remove_maximum(tweet)
    return tweet

# +++++++++++++++++++++++++++++++++++++++++++++
# WORDNET LEMMATIZER
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
# POS_TAGGER_FUNCTION : TYPE 1
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
    
def process_tweet_wordnet_lemmatizer( sentence ):
    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize( sentence )) 

    # we use our own pos_tagger function to make things simpler to understand.
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    # create lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:       
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)
    
    return lemmatized_sentence

# ===========================================================================================    
def decode_scores(score):
    return 1 if score>0.5 else 0

def predict_tweet(input):
    max_length = 50
    # PRE PROCESSING
    tw1 = process_tweet_phase1(str(input))
    tw2 = process_tweet_phase2(tw1)
    tw3 = process_tweet_wordnet_lemmatizer(tw2)
    # SEQUENCE
    tweet_tokenized_sequence = tokenizer.texts_to_sequences([tw3])

    # PADDING
    tweet_seq_pad = pad_sequences(tweet_tokenized_sequence, 
                                    maxlen=max_length, 
                                    padding='post')
    # PREDICTION
    y_pred_test = model.predict(tweet_seq_pad)
    y_pred_target = decode_scores(y_pred_test)

    if y_pred_target == 1:
        output = 'Positif'
    elif y_pred_target == 0 :
        output = 'Negatif'
    else :
        output = 'An error as occured'
     
    return output     


def run():
    st.title('Tweet Sentiment Prediction')

    # taking user inputs
    st.write("Type your tweet below")
    tweet = st.text_area(label="Enter tweet")
  
    # when user click on button it wille fetch the API
    predict_btn = st.button('Prédire')

    if predict_btn :    

        output = predict_tweet(str(tweet))

        if output is None:
            #raise Exception("Request failed with status {}, {}".format(response.status_code, response.text))
            response = 'Something went wrong, failed, not working...'
            with st.spinner('Classifying, please wait....'):
                        st.write(response)
        
        else :
            with st.spinner('Classifying, please wait....'):
                        st.write('Le sentiment du tweet est :', output)


if __name__ == '__main__':
	run()


