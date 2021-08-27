#!/c/Users/marti/AppData/Local/Programs/Python/Python39/python

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import ldamodel
from gensim.models.coherencemodel import CoherenceModel
import re
import glob
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
import os
from datetime import datetime
import json as js
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, SnowballStemmer,PorterStemmer
from nltk.stem.porter import*
import numpy as np
np.random.seed(2021)
wnl = WordNetLemmatizer()
stemmer = PorterStemmer()


def topic_modeling(path_to_cleaned_txt_file, output_path, output_filename, tfidf = False, max_num_topics = 50, step_size = 5, No_below = 0.3, No_above = 0.9):
  '''
  path_to_cleaned_txt_file: specify the path to the text file
  output_path: specify the path to save the output files
  output_filename: string name for the output file
  max_num_topics: number of topics
  step_size: incremental size from 2 up to maximum number of topics
  No_below: Filter out tokens that appear in less than no_below documents (absolute number) 
  No_above: Filter out tokens that appear in more than no_above documents (fraction of total corpus size, not absolute number)

  '''
  with open(path_to_cleaned_txt_file, 'r') as file:
    processed_docs = file.readlines()
    processed_docs = [i.strip().split() for i in processed_docs]
    processed_docs = [[i for i in j if not i in stop_words] for j in processed_docs]
  #Bag of words on the dataset
  dictionary = gensim.corpora.Dictionary(processed_docs)
    
  #gensim filter_out extremes
  dictionary.filter_extremes(no_below=No_below, no_above=No_above, keep_n=1000)

  #Gensim doc2bow
  bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

  #TF-IDF
  tfidf = gensim.models.TfidfModel(bow_corpus)
  corpus_tfidf = tfidf[bow_corpus]
  
  #Running LDA using Bag of Words
  
  #finding the optimal number of topics
  model_list = []
  c_v_values = []
  
  for num_topic in np.arange(2,max_num_topics,step_size):
    lda_model = gensim.models.ldamodel.LdaModel(corpus = bow_corpus, num_topics = num_topic, id2word=dictionary)
    model_list.append(lda_model)
    lda_cv_coherence = CoherenceModel(model = lda_model, texts=processed_docs, corpus = bow_corpus, coherence ='c_v')
    c_v_values.append(lda_cv_coherence.get_coherence())


  if tfidf is True:    
    #Building LDA Variants
    NUM_TOPICS = [i for i,j in zip(np.arange(2, max_num_topics, step_size), c_v_values) if j == np.max(c_v_values)][0]
    lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus_tfidf, num_topics = NUM_TOPICS, id2word = dictionary)
  else:
    #Running LDA using TF-IDF
    lda_model = [i for i,j in zip(model_list, c_v_values) if j == np.max(c_v_values)][0]    

  #Assigning topics to texts
  topics = lda_model.show_topics(formatted = False, num_words = 20, num_topics = NUM_TOPICS)
  my_topics = []
  for i in topics:
      my_words = []
      for word,num in i[1]:
          my_words.append(word)
      my_topics.append(my_words)
  my_topics = [' '.join(wnl.lemmatize(word) for word in i) for i in my_topics]
  for num,topic in enumerate(my_topics):
      print('\ntopic%i :'%num, topic)

  MyFile= open(output_path +'{}'.format(str(output_filename))+ \
  '{}'.format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))+'.txt', 'w')
  for line in my_topics:
      MyFile.write(line)
      MyFile.write('\n')
  MyFile.close()
    