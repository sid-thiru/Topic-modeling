import re
import nltk
import csv
import string
import gensim
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import word_tokenize, pos_tag
from gensim import corpora, models, similarities
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from operator import itemgetter


def create_data(file):
    with open(file, encoding = 'utf8') as csv_file:
        reader = csv.reader(csv_file, delimiter = ",", quotechar = '"')
        title = []
        description = []
        leafnode = []
        for row in reader:
            title.append(row[0])
            description.append(row[1])
            leafnode.append(row[2])
        return title[1:], description[1:], leafnode[1:]


def clean_data(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    stops = stopwords.words('english')
    nonan = re.compile(r'[^a-zA-Z ]')
    output = []
    for i in range(len(text)):
        sentence = nonan.sub('', text[i])
        words = word_tokenize(sentence.lower())
        filtered_words = [w for w in words if not w.isdigit()]
        filtered_words = [w for w in filtered_words if not w in stops]
        filtered_words = [w for w in filtered_words if not w in string.punctuation]
        tags = pos_tag(filtered_words)
        cleaned = ''
        for word, tag in tags:
          if tag == 'NN' or tag == 'NNS' or tag == 'VBZ' or tag == 'JJ' or tag == 'RB' or tag == 'NNP' or tag == 'NNPS' or tag == 'RBR':
            cleaned = cleaned + wordnet_lemmatizer.lemmatize(word) + ' '
        output.append(cleaned.strip())
    return output


#create a set of training documents for topic modeling. This is done by collpasing sets of ten documents from within a class into a single documents
#title + description of each class is too small for it to be a document by itself
#so ten documents from a class are combined to form a larger document 
#there are roughly 200 training samples for each class, so this leads to roughly 20 'new' documents per class 
def create_corpus(data, y, labels):
    corpus = []
    corpus_labels = []
    for i in range(len(labels)):
        label = labels[i]
        doc = ''
        j = 0
        k = 0
        while j < len(data):
            if k < 10:
                if y[j] == label:
                    doc = doc + data[j] + ' '
                    k = k + 1
                    j = j + 1
                else:
                    j = j + 1
            else:
                k = 0
                corpus.append(doc)
                corpus_labels.append(label)
                doc = ''
    return corpus, corpus_labels



#build a model for each of the 209 classes
#it was found that, for the given amount of data and the variety within this data, a single LDA model is not able to sufficiently strong topics that can be used in classification
def build_models(data, data_labels, unique_labels, N = 30):
    models = []
    vocabularies = []
    LDA = gensim.models.ldamodel.LdaModel
    for i in range(len(unique_labels)):
        L = unique_labels[i]
        corpus = [word_tokenize(data[i]) for i in range(len(data)) if data_labels[i] == L]
        vocabulary = corpora.Dictionary(corpus)                    
        BOW = [vocabulary.doc2bow(doc) for doc in corpus]
        LDAmodel = LDA(BOW, num_topics = N, id2word = vocabulary, passes = 25, alpha = 'auto', minimum_probability = 0.01, random_state = 30)
        models.append(LDAmodel)
        vocabularies.append(vocabulary)
    return models, vocabularies
        

#return the predicted leaf node label for an unseen sample
#takes a single title + description as input
#finds the LDA topics for the input sample from each of the 209 models
#the hypothesis is that there will be a model with a very strong topic probability for the input sample, and this model will correspond to the input's Apparel class label
def get_predictions(unseen_data, label_names):
    LDAresults = []
    LDAresults_lengths = []
    fill = np.zeros((3,2))
    for i in range(len(label_names)):
        BOW = [vocabularies[i].doc2bow(doc) for doc in unseen_data]
        try:
            LDAresults.append(np.array(LDAmodels[i].get_document_topics(BOW))[0])
        except ValueError:
            LDAresults.append(fill)
    #find the lengths of the LDA results from 209 models
    for i in range(len(label_names)):    
        LDAresults_lengths.append(len(LDAresults[i]))
    size = min(LDAresults_lengths)
    weights = []
    nodes = []
    #retain only the LDA results of minimum lengths, and check which of those results has the strongest topic distribution
    for i in range(len(label_names)):
        if LDAresults_lengths[i] == size:
            weights.append(np.amax(LDAresults[i][:,1]))
            nodes.append(label_names[i])
    predicted_label = nodes[np.argmax(weights)]
    return predicted_label







#data loading and cleaning
file = 'C:\\Users\\Sid\\Documents\\Regalix\\appareldata2.csv'
title, desc, leaf = create_data(file)
title = clean_data(title)
desc = clean_data(desc)
combined = desc[:]
for i in range(len(combined)):
    combined[i] = title[i] + ' ' + desc[i]


#integer to class label mapping
label_names = list(set(leaf))
label_indexes = {}  
for i in range(len(label_names)):
    label_indexes[label_names[i]] = i


#train test split
x_train, x_test, y_train, y_test = train_test_split(combined, leaf, stratify = leaf, test_size = 0.25, random_state = 0)

#chunk the samples in x_train to get a corpus with 'larger' documents
corpus, corpus_labels = create_corpus(x_train, y_train, label_names)

#build 209 LDA models
LDAmodels, vocabularies = build_models(corpus, corpus_labels, label_names)

#get the predicted labels
predicted_labels = []
for i in range(len(x_test)):
    feed = [word_tokenize(x_test[i])]
    pred = get_predictions(feed, label_names)
    predicted_labels.append(pred)
    
