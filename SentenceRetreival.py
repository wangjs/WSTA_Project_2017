# Author: Umer
# Date: 5-May-2017


# This script copies most of the code from WSTA_N15_information_retrieval' IPython Notebook uploaded by the tutor
# In this script I implement the sentence reterival part of the project
# For each question, we identify the correct sentence from the article
# This is just the basic technique and yields about 59% accuracy on train set

from __future__ import print_function
import json
import nltk
from math import log

from math import log
from collections import defaultdict, Counter

with open('QA_train.json') as data_file:
    data = json.load(data_file)




def extract_term_freqs(doc):
    tfs = Counter()
    for token in nltk.word_tokenize(doc):
        if token not in stopwords: # 'in' and 'not in' operations are much faster over sets that lists
            tfs[stemmer.stem(token.lower())] += 1
    return tfs

def compute_doc_freqs(doc_term_freqs):
    dfs = Counter()
    for tfs in doc_term_freqs.values():
        for term in tfs.keys():
            dfs[term] += 1
    return dfs


def query_vsm(query, index, k=10):
    accumulator = Counter()
    for term in query:
        postings = index[term]
        for docid, weight in postings:
            accumulator[docid] += weight
    return accumulator.most_common(k)





correctSentence = 0
totalQuestions = 0
i = 0
for article in data:
    print("Computing Article: ",i,'/',len(data))
    i+= 1
    corpus = article['sentences']

    stopwords = set(nltk.corpus.stopwords.words('english')) # wrap in a set() (see below)
    stemmer = nltk.stem.PorterStemmer()

    doc_term_freqs = {}
    for sent in corpus:
        term_freqs = extract_term_freqs(sent)
        doc_term_freqs[corpus.index(sent)] = term_freqs
    M = len(doc_term_freqs)

    doc_freqs = compute_doc_freqs(doc_term_freqs)

    vsm_inverted_index = defaultdict(list)
    for docid, term_freqs in doc_term_freqs.items():
        N = sum(term_freqs.values())
        length = 0

        # find tf*idf values and accumulate sum of squares
        tfidf_values = []
        for term, count in term_freqs.items():
            tfidf = float(count) / N * log(M / float(doc_freqs[term]))
            tfidf_values.append((term, tfidf))
            length += tfidf ** 2

        # normalise documents by length and insert into index
        length = length ** 0.5
        for term, tfidf in tfidf_values:
            # note the inversion of the indexing, to be term -> (doc_id, score)
            if length != 0:
                vsm_inverted_index[term].append([docid, tfidf / length])

    # ensure posting lists are in sorted order (less important here cf above)
    for term, docids in vsm_inverted_index.items():
        docids.sort()



    for qa in article['qa']:
        query = ""
        for token in nltk.word_tokenize(qa['question']):
            if token not in stopwords:  # 'in' and 'not in' operations are much faster over sets that lists
                query = query + ' ' + token
        result = query_vsm([stemmer.stem(term.lower()) for term in query.split()], vsm_inverted_index)
        totalQuestions += 1
        if len(result) > 0:
            bestSentence = result[0][0]
            if qa['answer_sentence'] == bestSentence:
                correctSentence += 1


print("The accuracy on train set is", (correctSentence/float(totalQuestions)))

