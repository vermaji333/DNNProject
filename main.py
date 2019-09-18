# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:30:44 2019

@author: deepa
"""
#print("hii")

import nltk
#nltk.download()
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
with open("intents.json") as file:
	data=json.load(file)
    
try:
    #raise ValueError("")
    with open("data.pickle","rb") as f:
        words,labels,training,output=pickle.load(f)
except:
    
    words=[]
    labels=[]
    docs_x=[]
    docs_y=[]
    
    for intent in data["intents"]:
    	for pattern in intent["patterns"]:
    		wrds=nltk.word_tokenize(pattern)
    		words.extend(wrds)
    		docs_x.append(wrds)
    		docs_y.append(intent["tag"])
    
    	if intent["tag"] not in labels:
    		labels.append(intent["tag"])
    
    words=[stemmer.stem(w.lower()) for w in words if w != "?"]
    words=sorted(list(set(words)))
    
    labels=sorted(labels)
    
    training=[]
    output=[]
    
    out_empty=[0 for _ in range(len(labels))]
    
    for x,doc in enumerate(docs_x):
        bag=[]
        wrds=[stemmer.stem(w.lower()) for w in doc]
        for w in words: 
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        out_row=out_empty[:]
        out_row[labels.index(docs_y[x])]=1
        
        training.append(bag)
        output.append(out_row)

    training=numpy.array(training)
    output=numpy.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)

tensorflow.reset_default_graph()

network=tflearn.input_data(shape=[None,len(training[0])])
network=tflearn.fully_connected(network, 8)
network=tflearn.fully_connected(network, 8)
network=tflearn.fully_connected(network, len(output[0]),activation="softmax")
network=tflearn.regression(network)

model=tflearn.DNN(network)

try:
    #raise ValueError("")
    model.load("model.tflearn")
except:
    
    model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
    model.save("model.tflearn") 
    
def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(w.lower()) for w in s_words ]
    for sw in s_words:
        for i,w in enumerate(words):
            if w==sw:
                bag[i]=1
                
    return numpy.array(bag)

def chat():
    print("********Its a QnA bot for customercare service of a shop (type quit to stop !)********")
    while True:
        inp=input("You: ")
        if inp.lower()=="quit":
            break
        
        results=model.predict([bag_of_words(inp,words)])[0]
        result_index=numpy.argmax(results)
        #print(results)
        if results[result_index]>0.7:
            
            tag=labels[result_index]
            for tg in data["intents"]:
                if tg["tag"]==tag:
                    response=tg["responses"]
            print("Bot:",random.choice(response))
        else:
            print("Bot: i didn't get you please try something else!")
chat()        
                    
        
             

	
