#Data visualizations for NASA project

#Set working directory
setwd("C:/Users/Amir/OneDrive/Academic/DATA 670/OCIO_Policy_TXT")
dir()

#Packages already installed
#Load tm package
library(tm)

#Load documents into working directory
docs <- Corpus(DirSource("."))
summary(docs)

#Step 1: Pre-process data

#Remove white space
docs.cleaned <- tm_map(docs, content_transformer(stripWhitespace))
inspect(docs.cleaned[4])

#Remove punctuation
docs.cleaned <- tm_map(docs.cleaned, removePunctuation)
inspect(docs.cleaned[[4]])

#Remove numbers
docs.cleaned <- tm_map(docs.cleaned, removeNumbers)
inspect(docs.cleaned[[4]])

#Convert to lower case
docs.cleaned <- tm_map(docs.cleaned, tolower)
inspect(docs.cleaned[[4]])

# #Create copy of corpus for use as dictionary later in script
# docscopy <- docs

#Lemmatize words
library(textstem)
docs.cleaned <- lemmatize_words(docs.cleaned)
inspect(docs.cleaned[[4]])

#Remove basic stop words
docs.cleaned <- tm_map(docs.cleaned, removeWords, stopwords("English"))
inspect(docs.cleaned[[4]])

#Remove additional stop words
docs.cleaned <- tm_map(docs.cleaned, removeWords, stopwords("SMART"))
inspect(docs.cleaned[[4]])


#Remove self-defined stop words
docs.cleaned <- tm_map(docs.cleaned, removeWords, c("-","-","always","also","many","get","us","much","would","shall","may","across","nasa","nodis","npd","chg","hq","osf","version","national",
                                                    "center","program","agency","office","government","provide","policy","ensure","include","page","chapter","date","http","gsfc","gov","npr"))

#Remove white space
docs.completed <- tm_map(docs.cleaned, content_transformer(stripWhitespace))
inspect(docs.completed[4])

# #Step 2: Word frequencies
# 
# #Create and display document term matrix
# dtm <- DocumentTermMatrix(docs.completed)
# dtm
# 
# #Compute how many times each term appears in corpus
# freq <- colSums(as.matrix(dtm))
# 
# #Display term frequencies in alpha order
# print(freq)
# 
# #Display top 20 terms by frequency in numeric descending order
# freq <- sort(colSums(as.matrix(dtm)), decreasing=TRUE)
# head(freq, 20)
# 
# #Display terms that appear at least 10 times
# findFreqTerms(dtm, lowfreq = 30)
# 
# #Remove terms with sparsity of 0.25 or greater
# dtms <- removeSparseTerms(dtm, 0.25)
# dtms
# 
# #Show terms with sparsity of less than 25%
# dtms$dimnames$Terms
# 

# #Step 4: correlation plots
# 
# #Rgraphviz package installed
# 
# #Create correlations plot with randomly selected 15 words appearing at least 30 times in corpus
# #Create separate plots with minimum correlation of 0.5 and 0.6
# #Show strength of correlation through line width
# plot(dtm, terms=findFreqTerms(dtm, lowfreq=30)[1:15], corThreshold=0.5, weighting = T)
# plot(dtm, terms=findFreqTerms(dtm, lowfreq=30)[1:15], corThreshold=0.6, weighting = T)
# 
# 
# #Step 5: word frequencies plot
# 
# #Load ggplot2 package
# library(ggplot2)
# 
# #Build ordered frequency plot and store it in p variable
# #Minimum frequency = 100
# wf <- data.frame(word=names(freq), freq=freq)
# p <- ggplot(subset(wf, freq>100), aes(x = reorder(word, -freq), y = freq)) +
#   geom_bar(stat = "identity") + 
#   theme(axis.text.x=element_text(angle=45, hjust=1))
# p

#Step 6: word cloud

#Load wordcloud package
library(wordcloud)

# #Create document term matrix
# dtm <- DocumentTermMatrix(docs.completed)
# 
# #Compute how many times each term appears in corpus
# freq <- colSums(as.matrix(dtm))
# 
# #Create word cloud with terms appearing at least 30 times in all documents
# wordcloud(names(freq), freq, min.freq = 30)
# 
# #Remove words appearing in fewer than 20% of all documents
# dtms <- removeSparseTerms(dtm, 0.8)
# #Find word frequencies
# freq <- colSums(as.matrix(dtm))
# #Add color codes based on word frequency
# dark2 <- brewer.pal(6, "Dark2")
# 
# #Create color coded word cloud with max 100 terms
# wordcloud(names(freq), freq, max.words=100, rot.per=0.2, colors=dark2)

#Create word cloud using TF-IDFf method as measure of importance of words
#Create DTM with weights based on TF-IDF method
dtm_tfidf <- DocumentTermMatrix(docs.completed, control = list(weighting = weightTfIdf)) 
dtm_tfidf = removeSparseTerms(dtm_tfidf, 0.75) #Remove terms with sparsity over 0.75
#dtm_tfidf #Display DTM
freq = data.frame(sort(colSums(as.matrix(dtm_tfidf)), decreasing=TRUE)) #Find and sort word frequencies
#Create color coded word cloud with max 100 terms
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(1, "Dark2")) 

#Network of terms

library ("igraph")                        #Load the igraph package
tdm<-TermDocumentMatrix(docs.completed)    # Term document matrix
tdm <- removeSparseTerms(tdm, 0.4)        # Remove sparse terms
termDocMatrix <- as.matrix(tdm)            # Convert tdm to matrix

termDocMatrix[termDocMatrix>=1] <- 1       # Set non-zero entries to 1 (1=term present, 0=term absent)
#Term matrix tracks how many times the terms appear together
# %*% is a matrix multiplication operator, and t is a transpose
termMatrix <- termDocMatrix %*% t(termDocMatrix)   

g <- graph.adjacency(termMatrix, weighted=T, mode="undirected")
g <- simplify(g)       # Remove the self-relationships
# V(g) is a graph vertex
V(g)$label <- V(g)$name  # Label each vertex with a term
V(g)$degree <- degree(g)
set.seed(3952)

#Create graph
#plot(g, layout=layout.fruchterman.reingold(g), vertex.color="cyan")
#plot(g, layout=layout_with_gem(g), vertex.color="pink")
#plot(g, layout=layout_as_star(g), vertex.color="yellow", vertex.shape="square")
plot(g, layout=layout_on_sphere(g), vertex.color="magenta")
plot(g, layout=layout_randomly(g), vertex.size=10)
#plot(g, layout=layout_in_circle(g), vertex.color="pink", vertex.size=35)
#plot(g, layout=layout_nicely(g), vertex.color="plum", vertex.size=25)



