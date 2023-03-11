import numpy as np
import pandas as pd
import matplotlib.pylab as plt


from collections import Counter
def histogram_string(column_name, column_title = ""): 
    '''
    enter the column_name, (optional) column_title 
    returns a histogram on a string column
    '''
    counts = Counter(column_name)
    labels, values = zip(*counts.items())
    valSort = np.argsort(values)[::-1]
    labels = np.array(labels)[valSort]
    values = np.array(values)[valSort]
    indexes = np.arange(len(labels))
    plt.bar(indexes, values)
    plt.xticks(indexes, labels, rotation=90)
    plt.title(column_title)
    plt.show()
    
    
    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



def LR_model(data): 
    '''
    perform logistic regression on data. The x_data are all columns except recommended and y_data is the recommended column. 
    logistic regression parameters: C=1, random_state=42, solver='lbfgs', max_iter=1e4     
    
    returns the logistic regression performance.
    '''
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Recommended_IND']), 
                                                    data['Recommended_IND'], test_size=0.3, random_state=1)
    
    lr = LogisticRegression(C=1, random_state=1, solver='lbfgs', max_iter=1e4)
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)
    print(classification_report(y_test, predictions))
    
    label = ['not_recommended','recommended']
    return pd.DataFrame(confusion_matrix(y_test, predictions), index=label, columns=label)


def accuracy_LR(data, Features_Used = ""):     
    '''
    returns featured columns used, logistic regression accuracy on train and test set. 
    '''
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Recommended_IND']), 
                                                    data['Recommended_IND'], test_size=0.3, random_state=1)
    lr = LogisticRegression(C=1, random_state=1, solver='lbfgs', max_iter=1e4)
    lr.fit(X_train,y_train)
    lr_train = accuracy_score(y_train, lr.predict(X_train)).round(4)
    lr_test = accuracy_score(y_test, lr.predict(X_test)).round(4)
    return [Features_Used,lr_train,lr_test]




from tqdm import tqdm
from nltk.corpus import wordnet
import unicodedata
import contractions
import re
import nltk
stop_words = nltk.corpus.stopwords.words('english')
stop_words.remove('no')
stop_words.remove('not')
nltk.download('wordnet')
wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
wnl = nltk.stem.wordnet.WordNetLemmatizer()

def remove_repeated_characters(texts):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
            
    correct_texts = [replace(word) for word in texts]
    return correct_texts

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def normalize_corpus(news_articles): 
    '''
    returns normalized dataset.
    '''
    norm_articles = []
    for article in tqdm(news_articles):
        article = article.lower() #convert all letters to lowercase
        article = ''.join([numb for numb in article if not numb.isdigit()])#remove numbers
        article = remove_accented_chars(article)#remove accented characters
        article = contractions.fix(article)#fix contractions
        article = re.sub(r'[^a-zA-Z0-9\s]', ' ', article, flags=re.I|re.A)#remove special characters
        article = re.sub(' +', ' ', article)
        article = article.strip()
        
        article_tokens = [token.strip() for token in wtk.tokenize(article)] #article_tokens
        article_tokens = remove_repeated_characters(article_tokens) #remove repeated characters in word
        article_tokens = [wnl.lemmatize(token) for token in article_tokens if not token.isnumeric()]#lemmatize data
        article_tokens = [token for token in article_tokens if len(token) > 2] #remove 1 letter words
        article_tokens = [token for token in article_tokens if token not in stop_words] #remove stop words
        article_tokens = list(filter(None, article_tokens))
            
        article_tokens = ' '.join(article_tokens)
        norm_articles.append(article_tokens)
            
    return norm_articles


from operator import itemgetter
#BOW N-gram features
def compute_ngrams(sequence, n):
    return list(zip(*(sequence[index:] for index in range(n))))

def flatten_corpus(corpus):
    return ' '.join([document.strip() for document in corpus])

def get_top_ngrams(corpus, ngram_val=1, limit=5):#show n-gram based features onto text columns
    '''
    
    '''
    corpus = flatten_corpus(corpus)
    tokens = nltk.word_tokenize(corpus)
    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(), key=itemgetter(1), reverse=True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
   
    sorted_ngrams = [(' '.join(text), freq) for text, freq in sorted_ngrams]

    return sorted_ngrams



def get_unique_strings(string_array): 
    #sort strings alphabetically in order to label them correctly in the confusion matrix
    list_of_unique_string = []
    unique_string = set(string_array)
    for string_array in unique_string:
        list_of_unique_string.append(string_array)
    return list_of_unique_string


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def cv_evaluateClassifier(classifier, x_data, y_data, min_df=0):
    '''
    Count vectorizer based features on a given model. 
    Data is split test_size = 30%, random_state = 1
    Count vectorizer parameters: binary=False, ngram_range=(1,3), default min_df=0
    Enter the model or classifier you want to analyze, x_data, y_data, (optional) min_df
    
    returns the model's performance
    '''
    print("Count Vectorizer based features and {} Model".format(classifier),"\n")
    train_x, test_x, train_y, test_y= train_test_split(np.array(x_data), np.array(y_data), test_size=0.30, random_state=1)
    
    cv =  CountVectorizer(binary=False, min_df= min_df, ngram_range=(1,3))
    cv_train_features = cv.fit_transform(train_x)
    cv_test_features = cv.transform(test_x)
    classifier.fit(cv_train_features, train_y)
    mnb_bow_predictions = classifier.predict(cv_test_features)
    
    print(classification_report(test_y, mnb_bow_predictions))
   
    label = get_unique_strings(test_y)
    label = sorted(label)# confusion matrix
    return pd.DataFrame(confusion_matrix(test_y, mnb_bow_predictions), index=label, columns=label)


def tv_evaluateClassifier(classifier, x_data, y_data, min_df=0):#TF-IDF based features on given models returns its performance
    '''
    TF-IDF based features on a given model. 
    Data is split test_size = 30%, random_state = 1
    Count vectorizer parameters: use_idf=True, ngram_range=(1,3), sublinear_tf=True, default min_df=0
    Enter the model or classifier you want to analyze, x_data, y_data, (optional) min_df
    
    returns the model's performance
    '''
    print("TF-IDF based features and {} Model".format(classifier),"\n")
    train_x, test_x, train_y, test_y= train_test_split(np.array(x_data), np.array(y_data), test_size=0.30, random_state=1)
    
    tv = TfidfVectorizer(use_idf=True, min_df=min_df, ngram_range=(1,3), sublinear_tf=True)
    tv_train_features = tv.fit_transform(train_x)
    tv_test_features = tv.transform(test_x)

    classifier.fit(tv_train_features, train_y)# train model
    mnb_tfidf_predictions = classifier.predict(tv_test_features)# predict on test data
    
    print(classification_report(test_y, mnb_tfidf_predictions))
    
    label = get_unique_strings(test_y)
    label = sorted(label)# confusion matrix labels
    return pd.DataFrame(confusion_matrix(test_y, mnb_tfidf_predictions), index=label, columns=label)

def accuracy_score_BOW(classifier, x_data, y_data, Features_Used = "", min_df = 0):
    '''
    returns features used, the accuracy of count vectorizer and TF-IDF based features on the given model (train and test accuracy)
    '''
    
    train_x, test_x, train_y, test_y= train_test_split(np.array(x_data), np.array(y_data), test_size=0.30, random_state=1)
    
    cv =  CountVectorizer(binary=False, min_df=min_df, ngram_range=(1,3))
    cv_train_features = cv.fit_transform(train_x)
    cv_test_features = cv.transform(test_x)
    classifier.fit(cv_train_features, train_y)
    cv_train = accuracy_score(train_y, classifier.predict(cv_train_features)).round(4)
    cv_test = accuracy_score(test_y, classifier.predict(cv_test_features)).round(4)
    
    tv = TfidfVectorizer(use_idf=True, min_df=min_df, ngram_range=(1,3), sublinear_tf=True)
    tv_train_features = tv.fit_transform(train_x)
    tv_test_features = tv.transform(test_x)
    classifier.fit(tv_train_features, train_y)
    accuracy_score(train_y, classifier.predict(tv_train_features)).round(4)
    tv_train = accuracy_score(train_y, classifier.predict(tv_train_features)).round(4)
    tv_test = accuracy_score(test_y, classifier.predict(tv_test_features)).round(4)
    
    return [Features_Used, cv_train, cv_test, tv_train, tv_test]