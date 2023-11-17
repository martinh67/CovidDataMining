# import the libraries required for the program
import json_lines

'''
a file reader class used to read in the data from a json file
'''
class FileReader:

    # delcare initialistion method with file_path and counter arguments
    def __init__(self, file_path, article_number):

        # open the file
        with open(file_path) as f:

            # initialise the body_text to a list
            self.body_text = []

            # initialise the published_at list
            self.published_at = []

            # initialise the counter
            self.article_number = article_number

            # assign content variable to the reader
            all_content = json_lines.reader(f)

            # for each bit of content
            for content in all_content:

                # if the counter is greater than 0
                if self.article_number > 0:

                    # append the value of the body to the body_text list
                    self.body_text.append(content['body'])

                    # append the value of published_at to the list
                    self.published_at.append(content['published_at'])

                    # decrement the counter
                    self.article_number -= 1

                # if the counter is 0
                else:

                    # break from the loop
                    break


    # magic method for directory
    def __dir__(self):

        # return parameters and outputs of the file reader class
        return ['file_path', 'article_number', 'body_text', 'published_at']


    # magic method for call
    def __call__(self, file_path, article_number):

        # return that the file reader instance has been called
        return("FileReader instance called")


    # method to format the class as a string
    def __str__(self):

        # return a string
        return(f'counter is {self.counter}, body_text length is {len(self.body_text)}, published_at length is {len(self.published_at)}')


    # method to produce machine readable representation of a type
    def __repr__(self):

        # return a formatted string of the data
        return(f'{self.counter}: {self.body_text} {self.published_at}')
