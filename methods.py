# import the libraries required for the program
import string
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import heapq

'''
method to plot the data with a dataframe and cluster number as parameters
'''
def plot_data(df, k):

    # declare the text variable
    text = df.body_text.values

    # call vectorise method for the text
    X = vectorise(text)

    # declare the kmeans object with the number of clusters
    kmeans = KMeans(n_clusters = k)

    # declare the y prediciton which fits the prediciton
    y_predict = kmeans.fit_predict(X.toarray())

    # declare a column in the dataframe for the y predict
    df['y'] = y_predict

    # declare the tsne object
    tsne = TSNE(verbose = 1, perplexity = 10)

    # fit the tsne
    X_embedded = tsne.fit_transform(X.toarray())

    # set the size of the plot
    sns.set(rc = {"figure.figsize" : (15,15)})

    # set the colour palette for the plot
    palette = sns.hls_palette(len(set(y_predict)), l = .4, s = .9)

    # create a scatter plot
    ax = sns.scatterplot(x = X_embedded[:,0], y = X_embedded[:,1],
    hue = y_predict, legend = "full", palette = palette)

    # add centroids to plot
    ax = sns.scatterplot(x = kmeans.cluster_centers_[:, 0],
    y = kmeans.cluster_centers_[:, 1], marker = "x", color ='black',
    s = 70, legend = False, label='centroid', ax = ax)

    # add the title
    plt.title("Tsne with kmeans labels")

    # save the image
    plt.savefig("article_clusters")

    # name the legend and move it off of the plot
    plt.legend(title = "clusters", loc = 2, bbox_to_anchor = (1,1))

    # show the plot
    plt.show()

'''
method to demonstarte the bag of words model being applied to the dataset
with a dataframe and the number of words being passed to the method
'''
def bag_of_words(df, word_number, vector_number):

    # declare a dictionary to hold the word frequencies
    word_frequency = {}

    # declare an array to hold the sentence vectors
    sentence_vectors = []

    # for article in the text
    for article in df['body_text']:

        # variable to hold the tokenised article
        tokens = nltk.word_tokenize(article)

        # for each token in the tokeniser
        for token in tokens:

            # if the token is not in the dictionary
            if token not in word_frequency.keys():

                # initialise the token value in the dictionary
                word_frequency[token] = 1

            # otherwise
            else:

                # increment the token value in the dictionary
                word_frequency[token] += 1

    # declare a list with the most frequent words specified by the function
    most_frequent_words = heapq.nlargest(word_number,
    word_frequency, key = word_frequency.get)

    # for each sentence in the dataframe
    for sentence in df['body_text']:

        # tokenise the sentences
        sentence_tokens = nltk.word_tokenize(sentence)

        # create a new sentence vector list
        new_sentence_vectors = []

        # for each token in the most frequent words
        for token in most_frequent_words:

            # if the token is in the sentence tokens list
            if token in sentence_tokens:

                # append 1 to the array
                new_sentence_vectors.append(1)

            # otherwise
            else:

                # append 0 to the array
                new_sentence_vectors.append(0)

        # append the results to the array
        sentence_vectors.append(new_sentence_vectors)

    # make an array using numpy
    sentence_vectors = np.asarray(sentence_vectors)

    # add a title to the plot
    plt.title('Bag of Words Model')

    # add an x axis label
    plt.xlabel('word frequency')

    # add y axis label
    plt.ylabel('top words in document')

    # plot the vector specified by the vector_number parameter
    plt.plot(sentence_vectors[vector_number])

    # get the axes object.
    ax = plt.gca()

    # adjust the y axis scale
    ax.locator_params('y', nbins=10)

    # adjust the x axis scale
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # adjust the x axis scale
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    # adjust the y axis scale
    y = [0,1]

    # adjust the y axis scale.
    plt.yticks(np.arange(0, 1.5, 0.5))

    print(sentence_vectors)

    # show the plot
    plt.show()


'''
method to vectorise the text
'''
def vectorise(text):

    # initialise vectoriser
    vectoriser = TfidfVectorizer()

    # cast data to tfidf representation
    X  = vectoriser.fit_transform(text)

    # return the vectorised data
    return X



'''
method to create a vector space module from a document
'''
def create_vector_space_model(document):

    # variable to remove the symbols
    remove_symbols = string.punctuation

    # create a list of symbols that we do not want
    pattern = r"[{}]".format(remove_symbols)

    # replace the symbols with blank spaces and put the doc to lower case
    document = re.sub(pattern, "", document.strip().lower())

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', '', document)

    # additionall preprocessing step to remove empty spaces
    document = re.sub(r'\s+',' ',document)

    # additional preprocessing step to remove puncation
    document = re.sub(r'\W',' ',document)

    # create a new instance of the document
    document = document.split()

    # return the document
    return document


'''
method to remove stopwords
'''
def remove_stop_words(row):

    # declare the unique set of stop words
    stop = set(stopwords.words("english"))

    # list comprehension
    # for each of the words in the row check if it is in the stop do not include
    row = [word for word in row if word not in stop]

    # iterate over the row and for every instance
    row = " ".join(row)

    # return the row
    return row
