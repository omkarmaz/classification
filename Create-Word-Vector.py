# -*- coding: utf-8 -*-
# This program will only create the Word Vectors

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk as nl
import re
import sys
import gensim

vector_size = 64   # vector size of the words

stopstring_list =  ['Please describe the issue. Include the actual error message, if applicable.:',\  # any repeating string which does not add value to distiguish the text
                 'Please Describe Error Message:',\
                 'Please Describe Business Impact of the Problem:'\
                 'Please select your Urgency:',\
                 'Caller\’s Name:',\
                 'Call back number:',\
                 'Email Address:',\
                 'Related case \#s \(case history\):',\
                 'Location\, remote/hotel/office:',\
                 'User\’s Working Hours:',\
                 'Application Name:',\
                 'Number of Users Affected:',\
                 'What is the issue/problem:',\
                 'Error message \(if any\):',\
                 'When was the last time it worked properly?',\
                 'Have you changed your password recently:',\
                 'Troubleshooting steps \(detailed\):',\
                 'A. Describe the problem in your own words :',\
                 'B. Describe the cause:',\
                 'C. Describe Actions take to address the problem  :',\
                 ' D. Has user approved problem Solution :','A-Issue:','B-Cause of the Issue:',\
                 'Troubleshooting steps :']
i=0
j=0
No_of_Args = len(sys.argv)
if (No_of_Args == 3):
    inputfile = str(sys.argv[1])
    param = str(sys.argv[2])
else:
    print('Please provide inputfile , parameter(Sheet1, Sheet2, Sheet3)')   # just options to choose input sheet from an excel
    exit()
    
train_set = pd.ExcelFile(inputfile)
train_df = train_set.parse(param)    
train_df = train_df.where(pd.notnull(train_df), None)
train_df['Notes_upd'] = train_df['Notes']
for j in range(len(stopstring_list)):
    train_df['Notes_upd']= train_df.Notes_upd.str.replace(stopstring_list[j],'') 
train_df['Notes_upd']= train_df.Notes_upd.str.lower()
train_df['tokens'] = [nl.word_tokenize(sentences) for sentences in train_df.Notes_upd]

stemmer = nl.stem.snowball.SnowballStemmer("english")

new_tokens = []
for index,item in train_df.iterrows():
    token_words = [word for word in item['tokens']]
    #token_words = [stemmer.stem(word) for word in item['tokens']]  # stemmers can be used but note that "customer" can become "custom". Stop words filter can also be used
    list_temp = []
    for word in token_words:
        str_temp = re.sub("\(+\)+\:+\,+\-+|\+|\.\.\.+|\`+|\'+|\.+|\"+", "", word)    # remove some un-necessary characters
        str_temp = re.sub("\<.\>","",str_temp)
        if len(str_temp)>1:
            list_temp.append(str_temp)
    new_tokens.append(list_temp)
train_df['new_tokens']=new_tokens

modelw2v = gensim.models.Word2Vec(iter=10000, sg=0, min_count=1, size = vector_size, window=5, workers=1) #workers=1 will use only 1 CPU core
modelw2v.build_vocab(x for x in train_df['new_tokens'])
modelw2v.train([x for x in train_df['new_tokens']], total_words=modelw2v.corpus_count, epochs=modelw2v.iter)
w2v = dict(zip(modelw2v.wv.index2word, modelw2v.wv.syn0))
filename = "modelw2v_MySE-Incidents-"+ str(vector_size)+".model"

modelw2v.save(filename)  # now you can use the wordvectors in your ML programs using Text data

del train_df
del train_set
del w2v
del modelw2v
