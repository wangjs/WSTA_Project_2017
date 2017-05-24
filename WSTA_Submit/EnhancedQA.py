#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Needed above lines to specify string encodings for some foreign characters (pound symbol etc.)


# Author: Muhammad Umer Altaf
# Date: 13-May-2017
# updated:14 May

#Comments Updated: 24-May-2017


## Please RUN the script after you have built the classifier,
# follow following instructions:
# Run: GenerateQuestionLabels.py (This will generate the dataset for the classifier)
# Run: BuildQuestionClassifier.py (This will fit a model on the data set generated in previous step )
# Run: EnhancedQA.py (This file!)


########### This is the main Enhanced QA Engine
########### Main Difference b/w BaseQA and enhancement is the
########### Question Classifier
########### It computes anwers for all questions in the test set and generates a .csv file of the results
########### Then it prints the accuracy


# importing all packages
from __future__ import print_function
import json
import nltk
from math import log
from collections import defaultdict, Counter
import os
import string
import pickle  # For caching of results
from dateutil.parser import parse
from time import ctime
import re
from nltk import StanfordPOSTagger
from nltk.tag.stanford import StanfordNERTagger
import operator

#This code needs BuildQuestionClassifier.py to be present in the same directory
from BuildQuestionClassifier import *
import BuildQuestionClassifier




# The three methods below are used for the tf-idf similarity measures
##########  coppied from workbook as is

def extract_term_freqs(doc):
    tfs = Counter()
    for token in nltk.word_tokenize(doc):
        if token not in stopwords:  # 'in' and 'not in' operations are much faster over sets that lists
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
############ End of coppied code


#Or you can give "DEV" here provided that that dataset is available in the same directory
runOn = "Test"

#This switch can be used to use a relaxed evaluation matric which awards for partial matches and match in many possible answers
#function defined later on, also can refer report
#if False then will only give a score 1 if exact match with correct answer (Default in project)
relaxedEvaluationMetric = False


# printing start time of the script
# This script should not take more that 4 or 5 minutes
print("Start Time:", ctime())

# initializing taggers and modals from NLTK
stanford_NER_tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
stanford_POS_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
stemmer = nltk.stem.PorterStemmer()

# os.environ["STANFORD_MODELS"] = "/Users/umeraltaf/Desktop/QA_Project/StanfordNER"
# stanford_NER_tagger = StanfordNERTagger('/Users/umeraltaf/Desktop/QA_Project/StanfordNER/english.all.3class.distsim.crf.ser.gz','/Users/umeraltaf/Desktop/QA_Project/StanfordNER/stanford-ner.jar')
# stanford_POS_tagger = StanfordPOSTagger('/Users/umeraltaf/Desktop/QA_Project/StanfordNER/english-bidirectional-distsim.tagger','/Users/umeraltaf/Desktop/QA_Project/StanfordNER/stanford-postagger.jar')
# stemmer = nltk.stem.PorterStemmer()

##Some path declarations for the precomputed models
# This is the cache file that will store the precomputed best sentences and tags
# so that we dont have to tag each time we run this script
if runOn=="DEV":
    fname = "bestSentencesTaggedDev.bin"
else:
    fname = 'bestSentencesTaggedTest.bin'
QuestionModelPATH = "QuestionClassificationModelStanford.pickle"


##Defining some global variables here

# This variable will store all tagged most relevant sentences
NER_tagged = None
NER_tagged2 = None
NER_tagged3 = None
POSTaggedSents = []
QuestionPOSTagDict = {}

# getting the list of englist stopwords
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.remove(
    'the')  ## After the error analysis of the results I realised that many answers have these words i.e. The President
stopwords.remove('of')  ## So will not exclude these

stopwordsAll = set(nltk.corpus.stopwords.words('english'))

# getting list of english punctuation marks to clean out sentences
PunctuationExclude = set(string.punctuation)
# again after error analysis I realised that these are part of answers and help in NER too
# i.e 75%
PunctuationExclude.remove(',')
PunctuationExclude.remove('-')
PunctuationExclude.remove('.')
PunctuationExclude.remove('\'')
PunctuationExclude.remove('%')




# Load the Required Dataset
if runOn == "Test":
    with open('QA_test.json') as data_file:
        data = json.load(data_file)
else:
    with open('QA_dev.json') as data_file:
        data = json.load(data_file)

# Main code part
if not os.path.exists(fname):  # Check if we already computed the best candidate sentences and thier entity tags

    #structures to hold results
    bestSentence = {}
    allBestSentences = []
    allSecondBestSentencesText = []
    allSecondBestSentences = []
    allThirdBestSentencesText = []
    allThirdBestSentences = []
    allBestSentencesText = []
    allQuestionText = []

    articleNo = -1
    for article in data:
        articleNo += 1
        print("Computing Article: ", articleNo + 1, '/', len(data))
        corpus = article['sentences']
        doc_term_freqs = {}

        # Preprocessing the sententeces and initilizing the tf-idf parameters
        for sent in corpus:
            sent2 = ''.join(ch for ch in sent if ch not in PunctuationExclude)  #######
            sent2 = sent2.replace(",", " ,") #some string cleaning so to seperate entities from comma/. example (USA, is in west)
            sent2 = sent2.replace(".", " .")
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

        # now for each question we reterive a list of most relevent sentences
        questionNo = -1
        for qa in article['qa']:
            questionNo += 1
            query = ""
            questionText = qa['question']
            questionText = ''.join(ch for ch in questionText if ch not in PunctuationExclude)  ######
            questionText = questionText.replace(",", " ,")
            questionText = questionText.replace(".", " .")
            for token in nltk.word_tokenize(questionText):
                if token not in stopwords:  # 'in' and 'not in' operations are much faster over sets that lists
                    query = query + ' ' + token
            result = query_vsm([stemmer.stem(term.lower()) for term in query.split()], vsm_inverted_index)

            # Here we concat the top 6 sentences for each question and process the stop words etc
            # we combine these sentences in to groups so we can handle them separately
            if len(result) > 0:
                bestSentenceText = article['sentences'][result[0][0]]
                if len(result) > 1:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[1][0]]
                bestSentenceText = ''.join(ch for ch in bestSentenceText if ch not in PunctuationExclude)
                bestSentenceText = bestSentenceText.replace(",", " ,")
                bestSentenceText = bestSentenceText.replace(".", " .").split()
                bestSentenceTokensNoStopWords = []
                for i in range(0, len(bestSentenceText) - 1):
                    if i > 0:
                        if bestSentenceText[i] not in stopwords or bestSentenceText[i][0].isupper():
                            bestSentenceTokensNoStopWords.append(bestSentenceText[i])
                    else:
                        bestSentenceTokensNoStopWords.append(bestSentenceText[i])

                allBestSentences.append(bestSentenceTokensNoStopWords)
                allBestSentencesText.append(bestSentenceText)
                best = result[0][0]
                bestSentence[articleNo, questionNo] = best

            else:
                allBestSentences.append([])  # to preserve question sequence
                allBestSentencesText.append(" ")


            #Second Group consisting of top 3rd and 4th result
            if len(result) > 2:
                bestSentenceText = article['sentences'][result[2][0]]
                if len(result) > 3:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[3][0]]
                bestSentenceText = ''.join(ch for ch in bestSentenceText if ch not in PunctuationExclude)
                bestSentenceText = bestSentenceText.replace(",", " ,")
                bestSentenceText = bestSentenceText.replace(".", " .").split()
                bestSentenceTokensNoStopWords = []
                for i in range(0, len(bestSentenceText) - 1):
                    if i > 0:
                        if bestSentenceText[i] not in stopwords or bestSentenceText[i][0].isupper():
                            bestSentenceTokensNoStopWords.append(bestSentenceText[i])
                    else:
                        bestSentenceTokensNoStopWords.append(bestSentenceText[i])

                allSecondBestSentences.append(bestSentenceTokensNoStopWords)
                allSecondBestSentencesText.append(bestSentenceText)
                best = result[0][0]
                bestSentence[articleNo, questionNo] = best

            else:
                allSecondBestSentences.append([])  # to preserve question sequence
                allSecondBestSentencesText.append(" ")

            # Second Group consisting of top 5th and 6th result
            if len(result) > 4:
                bestSentenceText = article['sentences'][result[4][0]]  ############
                if len(result) > 5:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[5][0]]  #######
                # if len(result) > 5:
                #     bestSentenceText = bestSentenceText + " " + article['sentences'][result[5][0]] #######
                bestSentenceText = ''.join(ch for ch in bestSentenceText if ch not in PunctuationExclude)
                bestSentenceText = bestSentenceText.replace(",", " ,")
                bestSentenceText = bestSentenceText.replace(".", " .").split()
                bestSentenceTokensNoStopWords = []
                for i in range(0, len(bestSentenceText) - 1):
                    if i > 0:
                        if bestSentenceText[i] not in stopwords or bestSentenceText[i][0].isupper():
                            bestSentenceTokensNoStopWords.append(bestSentenceText[i])
                    else:
                        bestSentenceTokensNoStopWords.append(bestSentenceText[i])

                allThirdBestSentences.append(bestSentenceTokensNoStopWords)
                allThirdBestSentencesText.append(bestSentenceText)
                best = result[0][0]
                bestSentence[articleNo, questionNo] = best

            else:
                allThirdBestSentences.append([])  # to preserve question sequence
                allThirdBestSentencesText.append(" ")

            allQuestionText.append(qa['question'])  # saving questions too for later usage


    #POS tag all questions (this is needed for the classifier, which origianly used
    # nltk pos tagger but, that did not run on lab machine to to save time in the classification pipeline
    # we pre tag the questions )
    #This will take about 1-2 mins
    tokenizedX = []
    for i in range(len(allQuestionText)):
        tokenizedX.append(wordpunct_tokenize(allQuestionText[i]))
    print(len(tokenizedX))
    print("Starting Question POS tagging")
    POSTaggedSents = stanford_POS_tagger.tag_sents(tokenizedX)
    print(len(tokenizedX))
    print("Question POS tagging done")

    for i in range(len(allQuestionText)):
        QuestionPOSTagDict[allQuestionText[i]] = POSTaggedSents[i]

    #This is a hack to get the classifier use Stanford POS tags instead of nltk one
    # to get this working I had to update the variable in the other files scope which we have included above
    BuildQuestionClassifier.POSTagDict = QuestionPOSTagDict


    # Now computing NER and other tags for answer sentences
    print("Computing NER start at:", ctime())

    NER_tagged = stanford_NER_tagger.tag_sents(allBestSentences)
    print("NER Time 1:", ctime())
    NER_tagged2 = stanford_NER_tagger.tag_sents(allSecondBestSentences)

    print("NER Time 2:", ctime())

    NER_tagged3 = stanford_NER_tagger.tag_sents(allThirdBestSentences)

    print("NER Time 3:", ctime())
    print("NER Tagging Done, Now Saving Variables")


    # saving the computed NER tags and sentences
    f = open(fname, 'wb')  # 'wb' instead 'w' for binary file
    pickle.dump({'NER_tagged': NER_tagged,
                 'allBestSentencesText': allBestSentencesText,
                 'NER_tagged2': NER_tagged2,
                 'allSecondBestSentencesText': allSecondBestSentencesText,
                 'NER_tagged3': NER_tagged3,
                 'allThirdBestSentencesText': allThirdBestSentencesText,
                 'allQuestionText': allQuestionText,
                 'POSTagDict': QuestionPOSTagDict

                 }, f, -1)  # -1 specifies highest binary protocol
    f.close()
    print("NER Saved")


else:  # NER tagged found
    f = open(fname, 'rb')  # 'rb' for reading binary file
    # Loading saved variables
    allVars = pickle.load(f)
    NER_tagged = allVars['NER_tagged']
    allBestSentencesText = allVars['allBestSentencesText']
    NER_tagged2 = allVars['NER_tagged2']
    allSecondBestSentencesText = allVars['allSecondBestSentencesText']
    NER_tagged3 = allVars['NER_tagged3']
    allThirdBestSentencesText = allVars['allThirdBestSentencesText']
    allQuestionText = allVars['allQuestionText']
    BuildQuestionClassifier.POSTagDict = allVars['POSTagDict']
    f.close()
    print("All saved variables loaded")

# Some static lists to recognize NUMBER
wordNumbers = {
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
}

dateNumbers = {

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
    'december',
    'AD',
    'BC'
    # 'year'
    # 'years',
    # 'day',
    # 'days',
    # 'month',
    # 'months'
}




# Following methods check if a token is a number
def checkIfRomanNumeral(token):
    thousand = 'M{0,3}'
    hundred = '(C[MD]|D?C{0,3})'
    ten = '(X[CL]|L?X{0,3})'
    digit = '(I[VX]|V?I{0,3})'
    return bool(re.match(thousand + hundred + ten + digit + '$', token))


def is_number(s):  # A basic function to check if a word/token is a number or not
    try:
        float(s)
        return True
    except ValueError:
        # print(s)
        if s.lower() in wordNumbers or s.lower() in dateNumbers or checkIfRomanNumeral(s) or bool(
                re.match(ur'([£$€])(\d+(?:\.\d{2})?)', s)):
            return True
        elif s[len(s) - 1] == u'%' and is_number(s[:len(s) - 1]):
            return True
        else:
            return False

# Trying to add NUMBER entity and removing ORGANIZATION from all possible answer sents
initial = 0
for NER_SENTS in [NER_tagged, NER_tagged2, NER_tagged3]:
    for answerSent in NER_SENTS:
        for i in range(0, len(answerSent) - 1):
            # tagging all other entities i.e. starts with capital and not tagged by NER
            if (answerSent[i][1] == 'O' and i > 0 and len(answerSent[i][0]) > 0 and answerSent[i][0][
                0].isupper() and i > 0 and answerSent[i - 1][0][0] != '.'):
                answerSent[i] = (answerSent[i][0], u'OTHER')
            # print(token)
            # # Dis-regarding ORGINIZATION tag
            # if answerSent[i][1] == "ORGANIZATION":
            #     answerSent[i] = (answerSent[i][0], u'OTHER')
                # print("****", answerSent[i][1])
            if is_number(answerSent[i][0]):
                answerSent[i] = (answerSent[i][0], u'NUMBER')
            if (i > 0 and answerSent[i][0] != "," and answerSent[i][0][0] == "," and is_number(answerSent[i][0][1:]) and
                        answerSent[i - 1][1] == 'NUMBER'):
                answerSent[i - 1] = (answerSent[i - 1][0] + answerSent[i][0], u'NUMBER')

import sys
#classifying all questions via the classifier we trained
QuestionTypesFromModel = []
QuestionTypes = []
if not os.path.exists(QuestionModelPATH):
    print("Please supply valid Question Classification Model, file searched:",QuestionModelPATH )
    sys.exit(1)
else:
    with open(QuestionModelPATH, 'rb') as f:
        model = pickle.load(f)
    print(len(allQuestionText))
    QuestionTypesFromModel = model.predict(allQuestionText)
    for t in range(0, len(QuestionTypesFromModel)):
        if (model.classes[QuestionTypesFromModel[t]] == 'ORGANIZATION'):
            QuestionTypes.append("OTHER")
        else:
            QuestionTypes.append(model.classes[QuestionTypesFromModel[t]])


#helper function to check if a sentence is in sentence case i.e.( I Am The Master!)
def isSentCase(sent):
    for x in sent.split():
        if not x[0].isupper():
            return False
    return True


#This function tries to extact the required entity type form the sentence provided
def extractAnswer(questionType, taggedBestAnswerSent, answerSentText, guessOTHERtype, apply3rdRule):
    answerList = []
    t = 0
    # trying to find questionType entity in answer

    # We find all the entities of question type in the answer text, i.e. the first rule of answer filtering
    for t in range(0, len(taggedBestAnswerSent) - 1):
        guessedAnswerText = ""
        if taggedBestAnswerSent[t][1] == questionType:
            for l in range(t, len(taggedBestAnswerSent) - 1):
                if taggedBestAnswerSent[l][1] == questionType:
                    guessedAnswerText = guessedAnswerText + " " + taggedBestAnswerSent[l][0]
                else:
                    break
        if ('l' in vars() or 'l' in globals()):
            t = l + 1
        if guessedAnswerText != "":
            answerList.append(guessedAnswerText)  # collect all the candidate answers seen

    if (guessOTHERtype):
        allQTypesList = ["NUMBER", "PERSON","LOCATION", "OTHER"]
        # we didnt find any matching entity type so we will give OTHER entity as answer
        if (len(answerList) < 1):
            questionType = "OTHER"
            t = 0
            for t in range(0, len(taggedBestAnswerSent) - 1):
                guessedAnswerText = ""
                if taggedBestAnswerSent[t][1] in allQTypesList:
                    for l in range(t, len(taggedBestAnswerSent) - 1):
                        if taggedBestAnswerSent[l][1] == taggedBestAnswerSent[t][1]:
                            guessedAnswerText = guessedAnswerText + " " + taggedBestAnswerSent[l][0]
                        else:
                            break
                if ('l' in vars() or 'l' in globals()):
                    t = l + 1
                answerList.append(guessedAnswerText)  # collect all the candidate answers seen
                # print(question['question'])
                # print(answerList)
                # print("-----" + question['answer'])

    guessedAnswerText = ""
    filteredAnswers = []

    # Filter as per second rule, that if candidate answer is fully included in the question we disregard it
    for ans in answerList:
        if ans not in question['question']:
            filteredAnswers.append(ans)

    # Trying to apply 3rd rule, but currently not stable all is commented and we select the first candidate ansewer
    if (len(filteredAnswers) > 0):
        guessedAnswerText = filteredAnswers[0]
    else:
        if (len(answerList) > 0):
            guessedAnswerText = answerList[0]
        else:
            guessedAnswerText = ""

    if guessedAnswerText != "":
        guessedAnswerText = guessedAnswerText[1:]  # remove the first space
        # print(guessedAnswerText)

    # Switch for 3rd rule answer ranking
    if (apply3rdRule):
        stemmedAnswerSent = []
        for token in answerSentText.split():
            stemmedAnswerSent.append(stemmer.stem(token.lower()))
        # Do some lemmatization or stemming or base words
        questionTextWithoutStop = []
        for qWord in question['question'].split():
            if stemmer.stem(qWord.lower()) not in stopwordsAll and stemmer.stem(qWord.lower()) in stemmedAnswerSent:
                questionTextWithoutStop.append(qWord.lower())

        for k in range(1, len(stemmedAnswerSent) - 1):
            if stemmedAnswerSent[k][0] == ',' and len(stemmedAnswerSent[k]) > 1:
                stemmedAnswerSent[k - 1] = stemmedAnswerSent[k - 1] + stemmedAnswerSent[k]

        stemmedAnswerSent = " ".join(stemmedAnswerSent)

        filteredAnswers2 = []
        for bigAns in filteredAnswers:
            found = False
            for extacted in filteredAnswers2:
                if bigAns in extacted:
                    found = True
            if not found:
                filteredAnswers2.append(bigAns)

        # print(filteredAnswers2)

        openDistances = {}
        for possAns in filteredAnswers2:
            for openClassWord in questionTextWithoutStop:
                subAnsList = []
                for subAns in possAns[1:].split():
                    subAnsList.append(stemmer.stem(subAns))
                if len(subAns) > 0:
                    subAnsList = subAnsList[0]

                try:
                    currIndex1 = stemmedAnswerSent.index(stemmer.stem(subAns.lower()))
                    currIndex2 = stemmedAnswerSent.index(stemmer.stem(openClassWord.lower()))
                    currDist = abs(currIndex1 - currIndex2)
                    openDistances[possAns[1:], openClassWord] = currDist
                except ValueError:
                    break

        answerDist = {}
        for (ans, ques) in openDistances:
            if ans in answerDist:
                answerDist[ans] = answerDist[ans] + openDistances[ans, ques]
            else:
                answerDist[ans] = openDistances[ans, ques]
        sorted_Distances = sorted(answerDist.items(), key=operator.itemgetter(1))
        # minDist = 9999999999999999999
        # minAns = ""
        # for ans in answerDist:
        #     if answerDist[ans] < minDist:
        #         minAns = ans
        #         minDist = answerDist[ans]

        if(len(sorted_Distances)>0):
            guessedAnswerText = sorted_Distances[0][0]
            filteredAnswers = []
            for (answer,distance) in sorted_Distances:
                filteredAnswers.append(answer)

    # Cleaning the answers a little bit, got these via errror analysis
    PunctuationExclude = set(string.punctuation)
    PunctuationExclude.remove(',')
    PunctuationExclude.remove('-')
    PunctuationExclude.remove('.')
    PunctuationExclude.remove('\'')
    PunctuationExclude.remove('%')
    guessedAnswerText = ''.join(ch for ch in guessedAnswerText if ch not in PunctuationExclude)  ######

    #Trying to represent numeric answers as per the format observed in train set
    if (questionType == 'NUMBER' and '.' in guessedAnswerText):
        guessedAnswerText = guessedAnswerText.replace(" ", "")
    if (questionType == 'NUMBER' and '%' in guessedAnswerText):
        guessedAnswerText = guessedAnswerText.replace(" ", "")
        guessedAnswerText = guessedAnswerText[:guessedAnswerText.index('%') + 1]
    if (questionType == 'NUMBER' and (
                    'what year' in question["question"].lower() or 'which year' in question["question"].lower())):
        for ans in filteredAnswers:
            try:
                if (str(parse(ans, fuzzy=True).year) in ans):
                    guessedAnswerText = str(parse(ans, fuzzy=True).year)
                    break
                else:
                    guessedAnswerText = guessedAnswerText
                    continue
            except ValueError:
                guessedAnswerText = guessedAnswerText
                continue
    if (questionType == 'NUMBER' and 'million' in guessedAnswerText):
        if (guessedAnswerText[guessedAnswerText.index('million') - 1] != " "):
            guessedAnswerText = guessedAnswerText.replace("million", " million")
    if (questionType == 'NUMBER' and 'billion' in guessedAnswerText):
        if (guessedAnswerText[guessedAnswerText.index('billion') - 1] != " "):
            guessedAnswerText = guessedAnswerText.replace("billion", " billion")

    return guessedAnswerText, filteredAnswers




#following function gives score for each answer, if we use relaxed measure as
# suggested in report we give partial score to partial matches and also
# award partial points if we can find answer in a list of proposed answers
def getEvaluationScore(correctAnswer,proposedAnswer,proposedAnswerList):
    if correctAnswer == proposedAnswer:
        return 1
    if(relaxedEvaluationMetric):
        if (proposedAnswer.lower() in correctAnswer.lower() or  correctAnswer.lower() in proposedAnswer.lower()) and proposedAnswer.lower() not in stopwordsAll:
            return 0.75
        for possibleAnswer in proposedAnswerList:
            if(possibleAnswer == correctAnswer):
                return 1/proposedAnswerList.index(possibleAnswer)
            elif (possibleAnswer.lower() in correctAnswer.lower() or  correctAnswer.lower() in possibleAnswer.lower()) and proposedAnswer.lower() not in stopwordsAll:
                return 0.75*1/1+(proposedAnswerList.index(possibleAnswer))
        return 0
    else:
        return 0

correct = 0
blank = 0
if RunOn=="Test":
    #initializing the output .csv files
    outFile = open('outPutTestSetOLD.csv', 'w')
    print(("id" + ',' + "answer"), file=outFile)
i = -1  # index of our NER_TAGGED list (i.e. questions)
for article in data:
    for question in article['qa']:  # For all questions in the article,
                                    # but notice that we add all questions
                                    #  to the same list in the end, indexed by i
        i += 1
        taggedBestAnswerSent = NER_tagged[i]
        taggedSecondBestAnswer = NER_tagged2[i]
        taggedThirdBestAnswer = NER_tagged3[i]
        answerSentText = u" ".join(
            allBestSentencesText[i])  # as we have sentence in the form of token lists so we join it into single string
        secondAnswerSentText = u" ".join(allSecondBestSentencesText[
                                             i])  # as we have sentence in the form of token lists so we join it into single string
        thirdAnswerSentText = u" ".join(allThirdBestSentencesText[i])
        # questionType = classifyQuestion(question['question']) #guess the question type, based on words in the question text
        questionType = QuestionTypes[i]




        # (questionType,taggedBestAnswerSent,answerSentText,guessOTHERtype,apply3rdRule)
        guessedAnswerText, filteredAnswers = extractAnswer(questionType, taggedBestAnswerSent, answerSentText, False,
                                                           True)

        # our top most 2 candidate sentences did not give any answers so now we search in 3rd 4th and 5th sentence
        if (guessedAnswerText == "" or guessedAnswerText == " " or guessedAnswerText.lower() in stopwordsAll):
            guessedAnswerText, filteredAnswers = extractAnswer(questionType, taggedSecondBestAnswer,
                                                               secondAnswerSentText, False, True)

        if (guessedAnswerText == "" or guessedAnswerText == " " or guessedAnswerText.lower() in stopwordsAll):
            guessedAnswerText, filteredAnswers = extractAnswer(questionType, taggedBestAnswerSent, answerSentText, True,
                                                               False)
        if (guessedAnswerText == "" or guessedAnswerText == " " or guessedAnswerText.lower() in stopwordsAll):
            guessedAnswerText, filteredAnswers = extractAnswer(questionType, taggedSecondBestAnswer,
                                                               secondAnswerSentText, True, False)
        if (guessedAnswerText == "" or guessedAnswerText == " " or guessedAnswerText.lower() in stopwordsAll):
            guessedAnswerText, filteredAnswers = extractAnswer(questionType, taggedThirdBestAnswer, thirdAnswerSentText,
                                                               False, False)
        if (guessedAnswerText == "" or guessedAnswerText == " " or guessedAnswerText.lower() in stopwordsAll):
            guessedAnswerText, filteredAnswers = extractAnswer(questionType, taggedThirdBestAnswer, thirdAnswerSentText,
                                                               True, False)
        if (guessedAnswerText == "" or guessedAnswerText == " " or guessedAnswerText.lower() in stopwordsAll):
            blank += 1



        if runOn=="DEV":
            correct += getEvaluationScore(question["answer"], guessedAnswerText, filteredAnswers)
        else:
            guessedAnswerText = guessedAnswerText.replace('"', "")
            guessedAnswerText = guessedAnswerText.replace(',', "-COMMA-")
            print((str(question['id']) + ',' + guessedAnswerText.encode('ascii', 'ignore')), file=outFile)

print("All Answer Computation Time:", ctime())
print("Blank Answers",blank)
if RunOn=="Test":
    outFile.close()
if(runOn=="DEV"):
    print("Correct/Score: ",correct)
    print("Accuracy: ", correct/float(i))
print("Done")
