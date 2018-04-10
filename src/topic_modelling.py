#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 22:57:09 2018

@author: siva
"""

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim import corpora,models
import json
from collections import defaultdict

NO_OF_WORDS_IN_EACH_REVIEW_THRESHOLD = 3
NO_OF_SENTENCES_IN_EACH_REVIEW_THRESHOLD = 7
NO_OF_TOPICS = 2

YELP_DATASET_PATH = "/Users/siva/Documents/CS506/datasets/yelp_dataset/review.json"
SAVING_FILE_PATH = "reviews.json"

def read_restaurant_reviews(FILE_PATH):
    business_reviews = defaultdict(list)
    with open(FILE_PATH,"r") as f_json:
        for i,line in enumerate(f_json):
            review = json.loads(line)
            business_reviews[review["business_id"]].append(review["text"])
            if i % 1000000 == 0:
                print(i)
    return business_reviews

def create_topics_dictionary(business_reviews):
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words("english"))
    #stemmer = PorterStemmer()
    restaurants_with_topics = {}
    
    for i,each_business in enumerate(business_reviews):
        if i % 10 == 0 and i != 0:
            break
        if len(business_reviews[each_business]) > 100:
            reviews_with_topics = defaultdict(list)
            for each_review in business_reviews[each_business]:
                sentences_each_review = each_review.split(".")
                texts = []
                for each_sentence in sentences_each_review:
                    tokens = tokenizer.tokenize(each_sentence.replace("\n","").lower())
                    tokens = [e for e in tokens if e not in stop_words]
                    if len(tokens)>NO_OF_WORDS_IN_EACH_REVIEW_THRESHOLD:
                        texts.append(tokens)
                if len(texts) > NO_OF_SENTENCES_IN_EACH_REVIEW_THRESHOLD:
                    dictionary = corpora.Dictionary(texts)
                    corpus = [dictionary.doc2bow(text) for text in texts]
                    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=NO_OF_TOPICS, id2word = dictionary, passes=20)
                    topic = ldamodel.show_topics(num_topics = 1,num_words = 1,formatted = False)[0][1][0][0]
                    reviews_with_topics[topic].append(each_review)
            restaurants_with_topics[each_business] = reviews_with_topics
    return restaurants_with_topics
    
def save_topics_reviews(restaurants_with_topics):
    with open(SAVING_FILE_PATH,"w") as write_json:
        for each in restaurants_with_topics:
            json.dump(restaurants_with_topics[each],write_json)
        
def main():
    business_reviews = read_restaurant_reviews(YELP_DATASET_PATH)
    topics_reviews = create_topics_dictionary(business_reviews)
    save_topics_reviews(topics_reviews)
    
if __name__ == "__main__":
    main()

           
    