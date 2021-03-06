
# Author: Muhammad Umer Altaf
# Date: 14-May-2017
# updated:14 May


# follwoing code is adapted from following resource
# Originally it was for sentiment analysis, I used it to
# classify questions in to answer types
# mainly PERSON,ORGINIZATION,LOCATION,NUMBER,OTHER
# http://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
# https://gist.github.com/bbengfort/044682e76def583a12e6c09209c664a1

#This file will fit and save a linear model on given dataset
# and save it on path defined by PATH below
# NOTE!: QuestionLabelsData.json is needed to build the model

## Please RUN the script after you have generated the label set,
# follow following instructions:
# Run: GenerateQuestionLabels.py (This will generate the dataset for the classifier)
# Run: BuildQuestionClassifier.py (This will fit a model on the data set generated in previous step )
# Run: EnhancedQA.py


import os
import time
import string
import pickle

from operator import itemgetter

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts
from nltk import StanfordPOSTagger
import json
os.environ["STANFORD_MODELS"] = "/Users/umeraltaf/Desktop/QA_Project/StanfordNER"

# stanford_POS_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
stanford_POS_tagger = StanfordPOSTagger('/Users/umeraltaf/Desktop/QA_Project/StanfordNER/english-bidirectional-distsim.tagger','/Users/umeraltaf/Desktop/QA_Project/StanfordNER/stanford-postagger.jar')

def timeit(func):
    """
    Simple timing decorator
    """
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        delta  = time.time() - start
        return result, delta
    return wrapper


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms input data by using NLTK tokenization, lemmatization, and
    other normalization and filtering techniques.
    """

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        """
        Instantiates the preprocessor, which make load corpora, models, or do
        other time-intenstive NLTK data loading.
        """
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct      = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        """
        Fit simply returns self, no other information is needed.
        """
        return self

    def inverse_transform(self, X):
        """
        No inverse transformation
        """
        return X

    def transform(self, X):
        """
        Actually runs the preprocessing on each document.
        """
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        """
        Returns a normalized, lemmatized list of tokens from a document by
        applying segmentation (breaking into sentences), then word/punctuation
        tokenization, and finally part of speech tagging. It uses the part of
        speech tags to look up the lemma in WordNet, and returns the lowercase
        version of all the words, removing stopwords and punctuation.
        """
        # Break the document into sentences


        for sent in sent_tokenize(document):

            # # Break the sentence into part of speech tagged tokens
            # docList = None
            # if document in POSTagDict:
            #     docList = POSTagDict[document]
            # else:
            #     taggedDocList = stanford_POS_tagger.tag(document)

            # print(POSTagDict)
            for token, tag in POSTagDict[document]:
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If punctuation or stopword, ignore token and continue
                # if token in self.stopwords or all(char in self.punct for char in token):
                #     continue
                # If punctuation or stopword, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform much more accurate WordNet lemmatization.
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)



@timeit
def build_and_evaluate(X, y, classifier=SGDClassifier, outpath=None, verbose=True):
    """
    Builds a classifer for the given list of documents and targets in two
    stages: the first does a train/test split and prints a classifier report,
    the second rebuilds the model on the entire corpus and returns it for
    operationalization.

    X: a list or iterable of raw strings, each representing a document.
    y: a list or iterable of labels, which will be label encoded.

    Can specify the classifier to build with: if a class is specified then
    this will build the model with the Scikit-Learn defaults, if an instance
    is given, then it will be used directly in the build pipeline.

    If outpath is given, this function will write the model as a pickle.
    If verbose, this function will print out information to the command line.
    """

    @timeit
    def build(classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """
        if isinstance(classifier, type):
            classifier = classifier()

        model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
            ('classifier', classifier),
        ])

        model.fit(X, y)
        return model

    # Label encode the targets
    labels = LabelEncoder()
    y = labels.fit_transform(y)
    # print(labels.classes_)
    # Begin evaluation
    if verbose: print("Building for evaluation")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    model, secs = build(classifier, X_train, y_train)

    if verbose: print("Evaluation model fit in {:0.3f} seconds".format(secs))
    if verbose: print("Classification Report:\n")

    y_pred = model.predict(X_test)
    print(clsr(y_test, y_pred, target_names=labels.classes_))

    if verbose: print("Building complete model and saving ...")
    model, secs = build(classifier, X, y)
    model.labels_ = labels

    if verbose: print("Complete model fit in {:0.3f} seconds".format(secs))
    model.classes = labels.classes_
    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))
    return model


def show_most_informative_features(model, text=None, n=20):
    """
    Accepts a Pipeline with a classifer and a TfidfVectorizer and computes
    the n most informative features of the model. If text is given, then will
    compute the most informative features for classifying that text.

    Note that this function will only work on linear models with coefs_
    """
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = model.named_steps['vectorizer']
    classifier = model.named_steps['classifier']

    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {} model.".format(
                classifier.__class__.__name__
            )
        )

    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append("Classified as: {}".format(model.predict([text])))
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
        )

    return "\n".join(output)

POSTaggedSents=[]
POSTagDict = {}
if __name__ == "__main__":
    PATH = "QuestionClassificationModelStanford.pickle"

    if not os.path.exists(PATH):
        # Time to build the model
        with open('QuestionLabelsData.json') as data_file:
            data = json.load(data_file)



        questions = []
        labels = []

        # print(data[26025])
        for i in range(len(data)):
            questions.append((data[i][0]))
            labels.append(data[i][1])

        X = questions
        y = labels
        print(len(X))
        print(len(y))


        tokenizedX = []
        for i in range(len(X)):
            tokenizedX.append(wordpunct_tokenize(X[i]))
        print(len(tokenizedX))
        print("Starting POS tagging")
        POSTaggedSents =stanford_POS_tagger.tag_sents(tokenizedX)
        print(len(tokenizedX))
        print("POS tagging done")

        for i in range(len(X)):
            POSTagDict[X[i]] = POSTaggedSents[i]
            # print(X[i], "**", POSTaggedSents[i])
        #building a dictionary for faster referance of the POSTags


        model = build_and_evaluate(X,y, outpath=PATH)

    else:
        with open(PATH, 'rb') as f:
            model = pickle.load(f)
        print(model.classes)
        print("Model Already Found")
    # print(show_most_informative_features(model))
