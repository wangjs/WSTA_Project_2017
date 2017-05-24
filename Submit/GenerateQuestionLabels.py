# Author: Umer
# Date: 20-May-2017

##In this script I will try to predict the question type based on the answers in the train set

#importing all packages
from __future__ import print_function
import json
import nltk
from math import log
from collections import defaultdict, Counter
import os
import string
import pickle  #For caching of results
from dateutil.parser import parse
from time import ctime
import re
from nltk import StanfordPOSTagger
from nltk.tag.stanford import StanfordNERTagger



#Following methods check if a token is a number
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
        # print(s)
        if s.lower() in wordNumbers or s.lower() in dateNumbers or checkIfRomanNumeral(s):
            return True
        elif s[len(s)-1] == u'%' and is_number(s[:len(s)-1]):
            return True
        else:
            return False

#Some static lists to recognize NUMBER
wordNumbers ={
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











#initializing taggers and modals from NLTK
stanford_NER_tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
stemmer = nltk.stem.PorterStemmer()

NER_CacheFIle = "AllTaggedCorrectAnswers.bin"

#we use last 300 articles to as train set for the classifier
with open('QA_train.json') as data_file:
    data = json.load(data_file)[-300:]



#NER tagging all the correct sentences
if not os.path.exists(NER_CacheFIle):  #Check if we already computed the best candidate sentences and thier entity tags
    questionTypes =[]
    correctAnswerSents = []
    questions = []
    answers = []
    NER_TaggedAnswerSents =[]
    NER_TaggedAnswerText = []
    articleNo = -1
    print("Computing NER start at:", ctime())
    i=-1
    for article in data:
        articleNo+=1
        currentCorrectAnswerSents =[]
        currentCorrectAnswerText = []
        print("Reading Article: ", articleNo + 1, '/', len(data))
        for qa in article['qa']:
            i+=1
            currentCorrectAnswerSents.append(article['sentences'][qa['answer_sentence']].split())
            currentCorrectAnswerText.append(qa['answer'].split())
            questions.append(qa['question'])
            answers.append(qa['answer'])
        NER_TaggedAnswerSents = NER_TaggedAnswerSents + stanford_NER_tagger.tag_sents(currentCorrectAnswerSents)
        NER_TaggedAnswerText= NER_TaggedAnswerText + stanford_NER_tagger.tag_sents(currentCorrectAnswerText)
        correctAnswerSents.append(currentCorrectAnswerSents)

    print("Computing NER end at:", ctime())
    #saving the computed NER tags and sentences
    f = open(NER_CacheFIle, 'wb')  # 'wb' instead 'w' for binary file
    pickle.dump({'NER_TaggedAnswerSents':NER_TaggedAnswerSents,
                 'correctAnswerSents': correctAnswerSents,
                 'NER_TaggedAnswerText':NER_TaggedAnswerText,
                 'questions':questions,
                 'answers':answers
                 }, f, -1)  # -1 specifies highest binary protocol
    f.close()
else:
    f = open(NER_CacheFIle, 'rb')  # 'rb' for reading binary file
    # Loading saved variables
    allVars = pickle.load(f)
    # NER_TaggedAnswerSents = [item for sublist in allVars['NER_TaggedAnswerSents'] for item in sublist]
    # NER_TaggedAnswerText = [item for sublist in allVars['NER_TaggedAnswerText'] for item in sublist]
    NER_TaggedAnswerSents = allVars['NER_TaggedAnswerSents']
    NER_TaggedAnswerText = allVars['NER_TaggedAnswerText']
    correctAnswerSents = [item for sublist in allVars['correctAnswerSents'] for item in sublist]
    questions = allVars['questions']
    answers = allVars['answers']



    f.close()
    print("All saved variables loaded")



#Assigning NUMBER tags manually
for answerSent in NER_TaggedAnswerSents:
    for i in range(0, len(answerSent) - 1):
        # tagging all other entities i.e. starts with capital and not tagged by NER
        # print(answerSent[i],answerSent[i-1])
        if (answerSent[i][1] == 'O' and i > 0 and len(answerSent[i][0]) > 0 and answerSent[i][0][0].isupper() and i > 0 and len(answerSent[i - 1][0]) > 0 and  answerSent[i - 1][0][0] != '.'):
            answerSent[i] = (answerSent[i][0], u'OTHER')
        if is_number(answerSent[i][0]):
            answerSent[i] = (answerSent[i][0], u'NUMBER')




#this was to limit the count of other tags that we might get
# as tests showed a skew towards this tag, so that our classifier might not be very biased
requiredOtherTags = 9000

#collecting entitity type for each question.
x=0
labels = []
for i in range(0,len(questions)-1):
    foundTypes = []
    for token in answers[i].split():
        for tag in NER_TaggedAnswerSents[i]:
            if token in tag[0]:
                if(tag[1]!='O' and (tag[1]!='OTHER' or (tag[1]=='OTHER' and requiredOtherTags>0)) ):
                    if(tag[1]=='OTHER' and requiredOtherTags>0):
                        requiredOtherTags -= 1
                    if(tag[1] not in foundTypes):
                        foundTypes.append(tag[1])
    if(len(foundTypes) == 0):
        currentTaggedAnswer = NER_TaggedAnswerText[i]
        for token in currentTaggedAnswer:
            if is_number(token[0]):
                foundTypes.append(u'NUMBER')
                break
    if(len(foundTypes)>0):
        labels.append((questions[i],foundTypes[0]))


entityCounts={}
for question in labels:
    entityCounts[question[1]] = entityCounts.get(question[1],0) + 1

print(entityCounts)
print(len(labels))


#Saving data set
with open('QuestionLabelsData.json', 'w') as fp:
    json.dump(labels, fp)