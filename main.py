# import the libraries required for the program
import pandas as pd
import time
import string
import re
from nltk.corpus import stopwords
import nltk
import numpy as np
from FileReader import FileReader
from methods import plot_data, bag_of_words, vectorise
from methods import create_vector_space_model, remove_stop_words

# add in the additional steps for the nltk stopwords
nltk.download('stopwords')

# download the additional steps for bag of words
nltk.download('punkt')

# declare the list of unique stopwords
stop = set(stopwords.words("english"))

# start the timer for the program
start = time.time()

'''
the main method for the program that displays clustering for
the file read in
'''
def main():

    # declare the number of articles
    article_number = 1000

    # declare the file path of the json file
    file_path = r"/Users/martinhanna/Downloads/aylien-covid-news.jsonl"

    # generate the content of the articles by instantiting the FileReader class
    content = FileReader(file_path, article_number)

    print(content.body_text)

    # declare a list for all of the rows
    all_rows = []

    # declare a dataframe to hold the body of the text and the date
    df = pd.DataFrame({"body" : content.body_text,
    "date" : content.published_at})

    # for every row in the body property of the dataframe
    for row in df.body:

        # create a vector space model of that row
        row = create_vector_space_model(row)

        # append to the all rows list
        all_rows.append(row)

    # set the body text for all rows
    df['body_text'] = all_rows

    # call the remove stop words method
    df.body_text = df.body_text.apply(remove_stop_words)

    # plot the data with a number of clusters
    plot_data(df, 5)

    # un comment this code to run the bag of words model
    # bag_of_words(df, 10, 0)


# magic method to run the main function
if __name__ == '__main__':
    main()

# print the timing of the program
print("\n" + 50 * "#")
print(time.time() - start)
print(50 * "#" + "\n")
