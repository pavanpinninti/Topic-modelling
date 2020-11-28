# Topic-modelling
Run Topicmodelling_Newscategory.ipynb file 
The process followed is explained here https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925

Link to the dataset https://www.kaggle.com/rmisra/news-category-dataset


Topic Modelling is the task of using unsupervised learning to extract the main topics (represented as a set of words) that occur in a collection of documents.
We have used  Latent Dirichlet Allocation (LDA) as it is powerful and quick to run.

# Latent Dirichlet Allocation (LDA)

LDA is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions.
Each document is modeled as a multinomial distribution of topics and each topic is modeled as a multinomial distribution of words.
LDA assumes that the every chunk of text we feed into it will contain words that are somehow related. Therefore choosing the right corpus of data is crucial.
It also assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution.

Dataset Used - Dataset I used is News Catergory Dataset. https://www.kaggle.com/rmisra/news-category-dataset 

This dataset contains around 200k news headlines from the year 2012 to 2018 obtained from HuffPost.

# Extracting Topics using LDA in Python
The code is present in the Newscatergorydataset_building_model.ipynb file.
# Preprocessing the raw text
This involves the following:
Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
Words that have fewer than 3 characters are removed.
All stopwords are removed.
Words are lemmatized — words in third person are changed to first person and verbs in past and future tenses are changed into present.
Words are stemmed — words are reduced to their root form.
We use the NLTK and gensim libraries to perform the preprocessing
# sample code
def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
#Tokenize and lemmatize

  def preprocess(text):

    result=[]
    
    for token in gensim.utils.simple_preprocess(text) :
    
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
        
            result.append(lemmatize_stemming(token))
            
    return result

# Converting text to bag of words
Prior to topic modelling, we convert the tokenized and lemmatized text to a bag of words — which you can think of as a dictionary where the key is the word and value is the number of times that word occurs in the entire corpus.

dictionary = gensim.corpora.Dictionary(processed_docs)

We can further filter words that occur very few times or occur very frequently.
Now for each pre-processed document we use the dictionary object just created to convert that document into a bag of words. i.e for each document we create a dictionary reporting how many words and how many times those words appear.

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Running LDA (Training the model)
This is actually quite simple as we can use the gensim LDA model. We need to specify how many topics are there in the data set. Lets say we start with 20 unique topics. Num of passes is the number of training passes over the document.

lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 20, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2)
 # Printing the topics and words present with their relative weights 
 for idx, topic in lda_model.print_topics(-1):
 
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
                                

# Model Testing 
The code is present in the Topicmodelling_Newscategory.ipynb file

# Loading our model
The files are located in the same repo.

#loading dict and bow corpus

loaded_dict = corpora.Dictionary.load('/content/Topic-modelling/dictionary.dict')

corpus = corpora.MmCorpus('/content/Topic-modelling/bow_corpus.mm')

#loading our lda

lda_model = gensim.models.LdaModel.load('/content/Topic-modelling/lda_newscatergory')

# Data preprocessing step for the unseen document
bow_vector = loaded_dict.doc2bow(preprocess(test))

test = text in unseen document

# Running the model 

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):

    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
    
    break
    
 This outputs the topic which has the highest probability
