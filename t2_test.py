# Author: Umer
# Date: 5-May-2017


########### THIS is just a test script for me (i.e. Umer), i will use it to play around ideas
########### may not contain any logical stuff

from __future__ import print_function
import json
import nltk
from math import log

from math import log
from collections import defaultdict, Counter
import os


import pickle  #For caching of results

fname = 'bestSentencesTaggedTEST.bin'

NER_tagged = None


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

os.environ["STANFORD_MODELS"] = "/Users/umeraltaf/Desktop/QA_Project/StanfordNER"

from nltk.tag.stanford import StanfordNERTagger

st = StanfordNERTagger('/Users/umeraltaf/Desktop/QA_Project/StanfordNER/english.all.3class.distsim.crf.ser.gz','/Users/umeraltaf/Desktop/QA_Project/StanfordNER/stanford-ner.jar')

with open('QA_test.json') as data_file:
    data = json.load(data_file)

if not os.path.exists(fname):  #Check if we already computed the best candidate sentences and thier entity tags

    bestSentence = {}
    allBestSentences = []
    articleNo = -1
    for article in data:
        articleNo += 1
        print("Computing Article: ",articleNo+1,'/',len(data))
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


        questionNo = -1
        for qa in article['qa']:
            questionNo += 1
            query = ""
            for token in nltk.word_tokenize(qa['question']):
                if token not in stopwords:  # 'in' and 'not in' operations are much faster over sets that lists
                    query = query + ' ' + token
            result = query_vsm([stemmer.stem(term.lower()) for term in query.split()], vsm_inverted_index)
            if len(result) > 0:
                allBestSentences.append(article['sentences'][result[0][0]].split())
                best = result[0][0]
                bestSentence[articleNo,questionNo] = best

            else:
                allBestSentences.append([]) #to preserve question sequence




    NER_tagged = st.tag_sents(allBestSentences)

    f = open(fname, 'wb')  # 'wb' instead 'w' for binary file
    pickle.dump(NER_tagged, f, -1)  # -1 specifies highest binary protocol
    f.close()
    print("NER Saved")


else: #NER tagged found
    f = open(fname, 'rb')  # 'rb' for reading binary file
    NER_tagged = pickle.load(f)
    f.close()
    print("NER Loaded")


wordNumbers =[
'zero',
'one',
'two',
'three',
'four',
'five',
'six',
'seven',
'eight',
'nine',
'ten',
'twenty',
'thirty',
'forty',
'fifty',
'sixty',
'seventy',
'eighty',
'ninety',
'hundred',
'thousand',
'million',
'billion',
'trillion'

]

dateNumbers = [

    'monday',
    'tuesday',
    'wednesday',
    'thursday',
    'friday',
    'saturday',
    'sunday',
    'january',
    'february',
    'march',
    'april',
    'may',
    'june',
    'july',
    'august',
    'september',
    'october',
    'november',
    'december'
]


def is_number(s): #A basic function to check if a word/token is a number or not
    try:
        float(s)
        return True
    except ValueError:
        if s.lower() in wordNumbers or s.lower() in dateNumbers:
            return True
        else:
            return False




#Trying to add NUMBER entity and removing ORGANIZATION
# print(st.tag('Rami Eid 99 Paris is studying Vxasd at Stony Brook University in NY'.split()))
# print(NER_tagged[0])
for answerSent in NER_tagged:
    for i in range (0,len(answerSent)-1):
        # tagging all other entities i.e. starts with capital and not tagged by NER
        if (answerSent[i][1] == 'O' and i > 0 and len(answerSent[i][0]) > 0 and answerSent[i][0][0].isupper()):
            answerSent[i] = (answerSent[i][0], u'OTHER')
        # print(token)
        # Dis-regarding ORGINIZATION tag
        if answerSent[i][1] == "ORGANIZATION":
            answerSent[i] = (answerSent[i][0], u'OTHER')
            print("****", answerSent[i][1])
        if is_number(answerSent[i][0]):
            answerSent[i] = (answerSent[i][0], u'NUMBER')






# #Concatinating adjacent same tag entities
# for answerSent in NER_tagged:
#     CompactTagged = []
#     lastTag = None
#     for i in range(0,len(answerSent)-1):
#         for j in range (i+1,len(answerSent)-1):
#             if answerSent[i][1] == answerSent[j][1]:

# Build a simple question classifier based on type of wh word in question:
def classifyQuestion(question):
    if "where" in question.lower() or "which" in question.lower():
        return "LOCATION"
    elif "who" in question.lower():
        return "PERSON"
    elif "how many" in question.lower() or "number" in question.lower() or "count" in question.lower():
        return "NUMBER"
    elif "when" in question.lower() or "date" in question.lower():
        return "NUMBER"
    else:
        return "OTHER"



outFile = open('outPutTestSetOLD.csv', 'w')
print(("id" + ',' + "answer"), file=outFile)

correct = 0
i = -1 #index of our NER_TAGGED list (i.e. questions)
for article in data:
    for question in article['qa']:
        i+=1
        taggedBestAnswerSent = NER_tagged[i]
        print(question['question'])
        print(taggedBestAnswerSent)
        questionType  = classifyQuestion(question['question'])
        print(questionType)
        answerList = []

        #trying to find questionType entity in answer
        for t in range(0,len(taggedBestAnswerSent)-1):
            guessedAnswerText = ""
            if taggedBestAnswerSent[t][1] == questionType:
                for l in range(t,len(taggedBestAnswerSent)-1):
                    if taggedBestAnswerSent[l][1] == questionType:
                        guessedAnswerText = guessedAnswerText + " " + taggedBestAnswerSent[l][0]
                    else:
                        break
            if ('l' in vars() or 'l' in globals()):
                t = l+1
            answerList.append(guessedAnswerText)

        guessedAnswerText = ""
        filteredAnswers = []
        for ans in answerList:
            if ans not in question['question']:
                filteredAnswers.append(ans)

        if(len(filteredAnswers) > 0):
            guessedAnswerText = filteredAnswers[0]
        else:
            if (len(answerList) > 0):
                guessedAnswerText = answerList[0]


        if guessedAnswerText != "":
            guessedAnswerText = guessedAnswerText[1:] #remove the first space
            print(guessedAnswerText)
        guessedAnswerText = guessedAnswerText.replace('"', "")
        guessedAnswerText = guessedAnswerText.replace(',',"-COMMA-")
        print((str(question['id'])+ ',' + guessedAnswerText.encode('ascii', 'ignore')), file=outFile)

outFile.close()
