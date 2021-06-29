from __future__ import division
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import urllib.request
import time
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import unicodedata
import string

#-------------------------------------------

#Step 1: Scrape policy documents from NODIS Library

#Code returns all links from main link https://nodis3.gcfc.nasa.gov , then it creates a list, from
#The list only links with “displayDir” are appended to a list and it text scraped and saved into a df in excel.

#import os
from bs4 import BeautifulSoup
import urllib.request

url = 'https://nodis3.gsfc.nasa.gov/Rpt_current_directives.cfm'
resp = urllib.request.urlopen(url)
data_path = 'C:/Users/Amir/OneDrive/Academic/DATA 670/NASA_Policies'
soup = BeautifulSoup(resp, "lxml",from_encoding=resp.info().get_param('charset'))

#Extract links from NODIS Library directives page

links_List = [] #create an empty link list

for link in soup.find_all('a', href=True):
    links_List.append(link['href'])
    #print(link['href'])  # find every  a element that has an href attribute & append it to  list & print

matching = [s for s in links_List if "displayDir" in s ]# return all links that have “displayDir” in them

#Extract text from policy documents

text_List = [] #empty text list

for links_above in matching:
    url = 'https://nodis3.gsfc.nasa.gov/' + links_above # Concatenate with the main link 
    #print(url) #print them
    resp = urllib.request.urlopen(url)
    soup = BeautifulSoup(resp, "lxml", from_encoding=resp.info().get_param('charset'))
    if soup.find_all('p')!=[]:
        text_List.append(soup.text) #append the text to text list
    else:
        pass

#Extract titles from policy documents 

titles_List = []

for i in range(len(text_List)):
    p = re.compile(r'(?<=Subject:).*?(?=\n)', re.IGNORECASE)
    m = p.search(text_List[i])
    if m:
        title = m.group()
        titles_List.append(title)

policy_dictionary = dict(zip(titles_List, text_List))

with open(data_path + 'policy_dictionary.pickle', 'wb') as handle:
    pickle.dump(policy_dictionary, handle)

#-------------------------------------------------------

#Step 2: Preprocess text

def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word)
        lemmas.append(lemma)
    return lemmas

def tokenize(text): #Tokenize text
    tokens = nltk.word_tokenize(text)
    return tokens

def clean_text(raw_text):
    """
    Define function to clean  text:
        1) Remove HTML tags
        2) Remove non-alphanumeric characters
        3) Remove single letters
        4) Convert text to lower case
        5) Remove white space
        6) Tokenize words
        7) Lemmatize words
        8) Remove stop words
    Returns a string of the cleaned text
    """
    denoise_text = re.sub('<[^>]+>', ' ', raw_text)
    letters_only = re.sub('[^a-zA-Z]', ' ', denoise_text)
    words_only = re.sub(r'\b[a-zA-Z]\b', '', letters_only)
    words = words_only.lower()
    words = words.strip()
    words = tokenize(words)
    words = lemmatize_words(words)
    stop_words = stopwords.words("english")
    newStopWords = ["–","—","always","also","many","get","us","much","would","shall","may","across","nasa","nodis","npd","chg","hq","osf","version","national",
                    "center","program","agency","office","government","provide","policy","ensure","include","page","chapter","date","http","gsfc","gov","npr",
                    "request","individual","number","process","chief","cfr","matter","work","guideline","control","august","standard","delegate",
                    "legal","plan","official","proposed","directorate","planning","goals","current","analysis","associate","activity","procedure","matter",
                    "review","material","seq","report","management","use","support","requirement","administrator","authority","act","federal"]
    stop_words.extend(newStopWords)
    useful_words = [x for x in words if not x in stop_words]
    # Combine words into a paragraph again
    useful_words_string = ' '.join(useful_words)
    return(useful_words_string)

text_List_clean = list(map(clean_text, text_List))
clean_policy_dictionary = dict(zip(titles_List, text_List_clean))

with open(data_path + 'policy_dictionary_clean.pickle', 'wb') as handle:
    pickle.dump(clean_policy_dictionary, handle)

#Inspect cleaned document
with open(data_path + 'policy_dictionary_clean.pickle', 'rb') as handle:
    clean_policy_dictionary = pickle.load(handle)

print("Sample of cleaned policy dictionary: 'Authority to Enter into Cooperative Research and Development Agreements'")
print(clean_policy_dictionary[' Authority to Enter into Cooperative Research and Development Agreements'])

#-----------------------------------------------------------

#Step 3: Engineer new features

#Create bag of words matrix

with open(data_path + 'policy_dictionary.pickle', 'rb') as handle:
    clean_policy_dictionary = pickle.load(handle)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(clean_policy_dictionary.values())
print("Bag of words matrix details:",cv_matrix)

#Create TF-IDF matrix

##with open(data_path + 'policy_dictionary.pickle', 'rb') as handle:
##    clean_policy_dictionary = pickle.load(handle)
##
##lemmatizer = WordNetLemmatizer()
##
##def lemmatize_words(words_list, lemmatizer): #Lemmatize words
##    return [lemmatizer.lemmatize(word) for word in words_list]

##tfidf = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
##tfs = tfidf.fit_transform(clean_policy_dictionary.values())
##print("TF_IDF matrix details:", tfs)

#Define functions to save and load sparse matrix
from scipy.sparse import csr_matrix

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

#Save sparse matrix
save_sparse_csr(data_path + 'policy_cv_matrix.npz', cv_matrix)

#Find nearest neighbors based on cosine distance

#Initialize nearest neighbors class and fit bag of words matrix into it
from sklearn.neighbors import NearestNeighbors
model_cv = NearestNeighbors(metric='cosine', algorithm='auto')
model_cv.fit(cv_matrix)

#Define function to print k nearest neighbors for given policy
def print_nearest_neighbors(query_cv_matrix, full_policy_dictionary, knn_model, k):
    """
    Inputs: a query cv_matrix vector, the dictionary of policies, the knn model, and the number of neighbors
    Print the k nearest neighbors
    """
    distances, indices = knn_model.kneighbors(query_cv_matrix, n_neighbors = k+1)
    nearest_neighbors = [list(full_policy_dictionary.keys())[x] for x in indices.flatten()]
    
    for policy in range(len(nearest_neighbors)):
        if policy == 0:
            print("Query Policy: {0}\n".format(nearest_neighbors[policy]))
        else:
            print("{0}: {1}\n".format(policy, nearest_neighbors[policy]))

#Test nearest neighbors function
policy_id = np.random.choice(cv_matrix.shape[0])
print_nearest_neighbors(cv_matrix[policy_id], clean_policy_dictionary, model_cv, k=5)

policy_id = np.random.choice(cv_matrix.shape[0])
print_nearest_neighbors(cv_matrix[policy_id], clean_policy_dictionary, model_cv, k=5)

policy_id = np.random.choice(cv_matrix.shape[0])
print_nearest_neighbors(cv_matrix[policy_id], clean_policy_dictionary, model_cv, k=5)

policy_id = np.random.choice(cv_matrix.shape[0])
print_nearest_neighbors(cv_matrix[policy_id], clean_policy_dictionary, model_cv, k=5)

policy_id = np.random.choice(cv_matrix.shape[0])
print_nearest_neighbors(cv_matrix[policy_id], clean_policy_dictionary, model_cv, k=5)


#Create document-topic matrix

from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

lda = LatentDirichletAllocation(n_topics=12, max_iter=10000, random_state=0)
dt_matrix = lda.fit_transform(cv_matrix)
features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'])
print("First 10 rows of document-topic matrix:")
print(features[0:10])

#Create term-topic matrix to show topics and their weights

tt_matrix = lda.components_
vocab = cv.get_feature_names()
print("Topic-term matrix:")
for topic_weights in tt_matrix:
    topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
    topic = sorted(topic, key=lambda x: -x[1])
    topic = [item for item in topic if item[1] > 0.6]
    print(topic)
    print()

#------------------------------------------------------

#Step 5: Create cluster model

#Create k-means cluster model

from sklearn.cluster import KMeans
#tfs = load_sparse_csr(data_path + 'policy_tf_idf.npz')
k = 12
km = KMeans(n_clusters=k, init='k-means++', verbose=1)
km.fit(features)

#Review number of assignments per cluster 
import  matplotlib.pyplot as plt
#%matplotlib inline

plt.hist(km.labels_, bins=k)
plt.show()

#Create dictionary with clusters as keys and policies as values
with open(data_path + 'policy_dictionary_clean.pickle', 'rb') as handle:
    clean_policy_dictionary = pickle.load(handle)

cluster_assignments_dict = {}

for i in set(km.labels_):
    current_cluster_policies = [list(clean_policy_dictionary.keys())[x] for x in np.where(km.labels_ == i)[0]]
    cluster_assignments_dict[i] = current_cluster_policies

#Inspect cluster assignments
print("Assignments for cluster 2:", cluster_assignments_dict[1])
print("Assignments for cluster 4:", cluster_assignments_dict[3])
print("Assignments for cluster 6:", cluster_assignments_dict[5])

#---------------------------------------------------------

#Step 6: Evaluate model

#Perform silhuoette analysis on models

from sklearn.metrics import silhouette_score, silhouette_samples 
#import seaborn as sns; sns.set()
import matplotlib.cm as cm

##scores = []
##values = np.arange(2, 10)

#Estimate silhouette score for clustering model using Euclidean distance metric
score = silhouette_score(features, km.labels_,
metric = 'euclidean', sample_size = features.shape[0])

#Display number of clusters and silhouette score
print("\nNumber of clusters =", k)
print("The silhouette_score is :", score)
#scores.append(score)

#Create cluster plot with policies visualized as bag of words clusters using t-SNE

#Reduce dimensional space of bag of words matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
k = 12
cv_matrix_reduced = TruncatedSVD(n_components=k, random_state=0).fit_transform(cv_matrix)

#Create 2-D representation of bag of words matrix using t-SNE
cv_matrix_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(cv_matrix_reduced)
print("Print details to confirm 2-D representation of bag of words matrix using t-SNE was created:", cv_matrix_embedded.shape)

#Plot policies colored according to their k-means cluster assignments
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(frameon=False)
plt.setp(ax, xticks=(), yticks=())
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                wspace=0.0, hspace=0.0)
plt.scatter(cv_matrix_embedded[:, 0], cv_matrix_embedded[:, 1], marker="x", c = km.labels_)
plt.show()



      
