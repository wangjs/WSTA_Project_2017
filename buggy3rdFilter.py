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

import string
import pickle  #For caching of results

fname = 'bestSentencesTagged.bin'

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

with open('QA_dev.json') as data_file:
    data = json.load(data_file)

stopwords = set(nltk.corpus.stopwords.words('english'))  # wrap in a set() (see below) ############## Remove from below
stemmer = nltk.stem.PorterStemmer()  ########

PunctuationExclude = set(string.punctuation)############
set(string.punctuation).remove(',')
set(string.punctuation).remove('-')
print(PunctuationExclude)

allBestSentences = []
if not os.path.exists(fname):  #Check if we already computed the best candidate sentences and thier entity tags

    correctSentence = 0
    totalQuestions = 0
    bestSentence = {}

    articleNo = -1
    for article in data:
        articleNo += 1
        print("Computing Article: ",articleNo+1,'/',len(data))
        corpus = article['sentences']


        doc_term_freqs = {}
        for sent in corpus:
            sent2 = ''.join(ch for ch in sent if ch not in PunctuationExclude) #######

            term_freqs = extract_term_freqs(sent2)
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
            questionText = qa['question'] #########
            questionText = ''.join(ch for ch in questionText if ch not in PunctuationExclude) ######

            for token in nltk.word_tokenize(questionText):
                if token not in stopwords:  # 'in' and 'not in' operations are much faster over sets that lists
                    query = query + ' ' + token
            result = query_vsm([stemmer.stem(term.lower()) for term in query.split()], vsm_inverted_index)
            totalQuestions += 1
            if len(result) > 0:
                bestSentenceText = article['sentences'][result[0][0]]  ############

                if len(result) > 1:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[1][0]] #######
                if len(result) > 2:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[2][0]] #######
                # if len(result) > 3:
                #     bestSentenceText = bestSentenceText + " " + article['sentences'][result[3][0]] #######
                # if len(result) > 4:
                #     bestSentenceText = bestSentenceText + " " + article['sentences'][result[4][0]] #######
                bestSentenceText = ''.join(ch for ch in bestSentenceText if ch not in PunctuationExclude).split()



                bestSentenceTokensNoStopWords = []
                for i in range(0,len(bestSentenceText)-1):
                    if i >0:
                        if bestSentenceText[i] not in stopwords or bestSentenceText[i][0].isupper():
                            bestSentenceText[i] = bestSentenceText[i].replace(" ,", ",")
                            bestSentenceTokensNoStopWords.append(bestSentenceText[i])

                allBestSentences.append(bestSentenceTokensNoStopWords) #####################
                best = result[0][0]
                bestSentence[articleNo,questionNo] = best
                if qa['answer'] in bestSentenceText:
                    correctSentence += 1
            else:
                allBestSentences.append([]) #to preserve question sequence


    print("The accuracy on dev set is", (correctSentence/float(totalQuestions)))


    NER_tagged = st.tag_sents(allBestSentences)

    print('NER Tagging Done')




    f = open('bestSentencesTagged.bin', 'wb')  # 'wb' instead 'w' for binary file
    pickle.dump(NER_tagged, f, -1)  # -1 specifies highest binary protocol
    f.close()
    print("NER Saved")


else: #NER tagged found
    f = open('bestSentencesTagged.bin', 'rb')  # 'rb' for reading binary file
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
'eleven',
'twelve',
'thirteen'
'fourteen',
'fifteen'
'sixteen',
'eighteen'
'nineteen',
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
'trillion',
'byte'
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


locationList = [

'city',
    'country',
    'location',
    'continent',
    'state',
    'area',
    'river',
    'pond',
    'fall',
    'desert'

]


import re


def checkIfRomanNumeral(token):
    thousand = 'M{0,3}'
    hundred = '(C[MD]|D?C{0,3})'
    ten = '(X[CL]|L?X{0,3})'
    digit = '(I[VX]|V?I{0,3})'
    return bool(re.match(thousand + hundred + ten + digit + '$', token))


def is_number(s): #A basic function to check if a word/token is a number or not
    try:
        float(s)
        return True
    except ValueError:

        if s.lower() in wordNumbers or s.lower() in dateNumbers or checkIfRomanNumeral(s):
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
            # print("****", answerSent[i][1])
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
    if "where" in question.lower() or "which" in question.lower() or question.lower() in locationList :
        return "LOCATION"
    elif "who" in question.lower():
        return "PERSON"
    elif "how many" in question.lower() or "number" in question.lower() or "count" in question.lower():
        return "NUMBER"
    elif "when" in question.lower() or "date" in question.lower():
        return "NUMBER"
    else:
        return "OTHER"




from nltk import StanfordPOSTagger
os.environ["STANFORD_MODELS"] = "/Users/umeraltaf/Desktop/QA_Project/StanfordNER"
stanford_tagger = StanfordPOSTagger('/Users/umeraltaf/Desktop/QA_Project/StanfordNER/english-bidirectional-distsim.tagger','/Users/umeraltaf/Desktop/QA_Project/StanfordNER/stanford-postagger.jar')




correct = 0
possCorrect = 0
wrongNumber = 0
totalans = 0
multiAnswer = 0
i = -1 #index of our NER_TAGGED list (i.e. questions)
for article in data:
    for question in article['qa']:

        i+=1
        taggedBestAnswerSent = NER_tagged[i]
        # answerSentText = allBestSentences[i]     ##############
        questionType = classifyQuestion(question['question'])
        answerList = []

        #trying to find questionType entity in answer
        for t in range(0,len(taggedBestAnswerSent)-1):
            guessedAnswerText = ""
            if taggedBestAnswerSent[t][1] == questionType :
                for l in range(t,len(taggedBestAnswerSent)-1):
                    if taggedBestAnswerSent[l][1] == questionType :
                        guessedAnswerText = guessedAnswerText + " " + taggedBestAnswerSent[l][0]
                    else:
                        break
            if('l' in vars() or 'l' in globals()):
                t = l+1
            answerList.append(guessedAnswerText)

        guessedAnswerText = ""
        filteredAnswers = []
        for ans in answerList:
            if ans not in question['question']:
                filteredAnswers.append(ans)

        if (len(filteredAnswers) > 0):
            totalans += len(filteredAnswers)
            multiAnswer += 1
            guessedAnswerText = filteredAnswers[0]


        # ############ No we have first 2 rules applied, now applying third rule, i.e. closest to closed class word in question
        # #### http://universaldependencies.org/u/pos/
        #
        #
        # openClassTags = {
        #     'JJ',
        #     'JJR',
        #     'JJS',
        #     'NN',
        #     'NNP',
        #     'NNPS',
        #     'NNS',
        #     'RB',
        #     'RBR',
        #     'RBS',
        #     'UH',
        #     'VB',
        #     'VBD',
        #     'VBG',
        #     'VBN',
        #     'VBP',
        #     'VBZ'
        #
        #
        # }
        #
        #
        # posTaggedQuestion = stanford_tagger.tag(question['question'])
        # posTaggedAnswer = stanford_tagger.tag(answerSentText)
        #
        # openClassInQuestion = []
        # openClassInAnswer = []
        #
        #
        #
        # ## remove all closed class words:
        # for tag in posTaggedQuestion:
        #     if tag[1] in openClassTags:
        #         openClassInQuestion.append(tag[0])
        #
        # for tag in posTaggedAnswer:
        #     if tag[1] in openClassTags:
        #         openClassInAnswer.append(tag[0])
        #
        #
        # ## Find same in both
        # setSame = set(openClassInAnswer) & set(openClassInQuestion)
        #
        # distancesFromOpenClass={}
        #
        # for possibleAnswer in filteredAnswers:
        #     for questionWord in setSame:
        #         distancesFromOpenClass[possibleAnswer[1:],questionWord] = answerSentText.index(possibleAnswer[1:])
        #
        #
        # distanceAnswer = {}
        # for A,Q in distancesFromOpenClass:
        #     if A in distanceAnswer:
        #         distanceAnswer[A] += distancesFromOpenClass[A,Q]
        #     else:
        #         distanceAnswer[A] = distancesFromOpenClass[A,Q]
        #
        #
        # minDist = 1000000000
        # minA = ""
        # for A in distanceAnswer:
        #     if distanceAnswer[A] <= minDist:
        #         minDist = distanceAnswer[A]
        #         minA = A
        #
        # guessedAnswerText  = minA
        #
        ######## End today work


        for ansC in filteredAnswers:
            if ansC[1:] == question['answer'] and filteredAnswers[0]!= ansC:
                # print(question["question"],"::",NER_tagged[i])
                # print(questionType, "::", filteredAnswers, question["answer"])
                possCorrect+=1
                break
        else:
            if (len(answerList) > 0):
                guessedAnswerText = answerList[0]
        PunctuationExclude = set(string.punctuation)
        set(string.punctuation).remove('-')
        guessedAnswerText = ''.join(ch for ch in guessedAnswerText if ch not in PunctuationExclude)  ######
        if guessedAnswerText != "" and guessedAnswerText[0] == ' ':
            guessedAnswerText = guessedAnswerText[1:]  # remove the first space
            # print(guessedAnswerText)


        if guessedAnswerText == question['answer']:
            correct +=1

        elif questionType == 'NUMBER':
            wrongNumber += 1
            # print(question['question'])
            # print(taggedBestAnswerSent)
            # print(questionType)
            # print(guessedAnswerText)
            # print("-----" + question['answer'])
        print(i / float(8460), ":", correct)


print("wrong in selected cat",wrongNumber)
print("total",i)
print("correct",correct)
print("correct in multi ans",possCorrect)
print("avg multi ans len", totalans/float(multiAnswer))
print(multiAnswer)




print(stanford_tagger.tag("Crazy monkey jumping on the tree"))