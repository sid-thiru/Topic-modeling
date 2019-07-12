# EXPERIMENTS WITH GENSIM

Topic models can be used to create features for text classification problems. A topic vector (a vector containing the probabilities of each topic being in a given document) can be the input to a classifier. Usually the topic vectors are used in addition to other features, such as TFIDF. 

Note: The gensim python script contains code for a slightly different approach - which is to build an individual LDA model for each class label (so in this case, 209 LDA models). This approach was not able to match the performance of the approach described here. The code on the script can be easily modified to take the approach described here.

### DATA PRE-PROCESSING
* Each product is represented by its 'Title' and 'Description'. There are 209 classes into which a given product can be classified. This is not a multilabel classification problem, so each product has to be assigned one out of the 209 classes only. The dataset contains roughly 200 samples per class
* The data was cleaned by removing stopwords, punctuations and special characters from the text
* In this dataset, a product's Title + Description is too small for it to be a input document to an LDA model. This issue is handled by collpasing sets of ten documents from within a class into a single document. There are roughly 200 training samples for each class, so this leads to 20 'new & larger' documents per class 


### FEATURE EXTRACTION	
* Gensim's LDA model requires the training corpus to be in a Bag-of-words representation. This can easily be done using Gensim's doc2bow function


### TOPIC MODELS - Using topic vectors to represent documents
* A LDA model (that generates 200 topics) was trained on the corpus. There are roughly 20 documents per class label, so we obtain 20 topic vectors per class. A topic vector is a vector of length 200, with each value representing the probability of a topic being in that document
* For a new unseen document, the LDA model that was previously trained is used to generate a topic vector
* The cosine similarity of this topic vector, and the topic vectors generated on the training data, is computed.  A majority vote is done among the top ten closest training topic vectors, and the majority label is thus assigned to the unseen document
* This approach was able to classify at 50% precision
	
