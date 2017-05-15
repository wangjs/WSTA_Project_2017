# Author: Umer
# Date: 5-May-2017
# updated:14 May

########### This is the main BASE QA Engine
########### It computes anwers for all questions in the first 100 articles of the dev set
########### Then it prints the accuracy


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


runOn = "Train"


#The three methods below are used for the tf-idf similarity measures
##########  coppied from workbook

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

############ End of coppied code




#printing start time of the script
print("Start Time:",ctime())

#initializing taggers and modals from NLTK
os.environ["STANFORD_MODELS"] = "/Users/umeraltaf/Desktop/QA_Project/StanfordNER"
stanford_NER_tagger = StanfordNERTagger('/Users/umeraltaf/Desktop/QA_Project/StanfordNER/english.all.3class.distsim.crf.ser.gz','/Users/umeraltaf/Desktop/QA_Project/StanfordNER/stanford-ner.jar')
stanford_POS_tagger = StanfordPOSTagger('/Users/umeraltaf/Desktop/QA_Project/StanfordNER/english-bidirectional-distsim.tagger','/Users/umeraltaf/Desktop/QA_Project/StanfordNER/stanford-postagger.jar')
stemmer = nltk.stem.PorterStemmer()



#This is the cache file that will store the precomputed best sentences and tags
#so that we dont have to tag each time we run this script
if(runOn=="DEV"):
    fname = 'bestSentencesTaggedDev.bin'
else:
    fname = 'bestSentencesTaggedTrain.bin'


#This variable will store all tagged most relevant sentences
NER_tagged = None




#Load the dataset, note that as train set is large I only load first 50 articles

if(runOn == "DEV"):
    with open('QA_dev.json') as data_file:
        data = json.load(data_file)
else:
    with open('QA_train.json') as data_file:
        data = json.load(data_file)[:50]



# getting the list of englist stopwords
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.remove('the') ## After the error analysis of the results I realised that many answers have these words i.e. The President
stopwords.remove('of') ## So will not exclude these





# getting list of english punctuation marks to clean out sentences
PunctuationExclude = set(string.punctuation)
#again after error analysis I realised that these are part of answers and help in NER too
# i.e 75%
PunctuationExclude.remove(',')
PunctuationExclude.remove('-')
PunctuationExclude.remove('.')
PunctuationExclude.remove('\'')
PunctuationExclude.remove('%')


#Main code part
if not os.path.exists(fname):  #Check if we already computed the best candidate sentences and thier entity tags
    correctSentence = 0 #just to compute statistics on train set
    totalQuestions = 0
    bestSentence = {}
    allBestSentences = []
    allBestSentencesText = []
    allQuestionText = []

    articleNo = -1
    for article in data:
        articleNo += 1
        print("Computing Article: ",articleNo+1,'/',len(data))
        corpus = article['sentences']
        doc_term_freqs = {}

        #Preprocessing the sententeces and initilizing the tf-idf parameters
        for sent in corpus:
            sent2 = ''.join(ch for ch in sent if ch not in PunctuationExclude) #######
            sent2=sent2.replace(",", " ,")
            sent2=sent2.replace(".", " .")
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

        #now for each question we reterive a list of most relevent sentences
        questionNo = -1
        for qa in article['qa']:
            questionNo += 1
            query = ""
            questionText = qa['question']
            questionText = ''.join(ch for ch in questionText if ch not in PunctuationExclude) ######
            questionText = questionText.replace(",", " ,")
            questionText = questionText.replace(".", " .")
            for token in nltk.word_tokenize(questionText):
                if token not in stopwords:  # 'in' and 'not in' operations are much faster over sets that lists
                    query = query + ' ' + token
            result = query_vsm([stemmer.stem(term.lower()) for term in query.split()], vsm_inverted_index)
            totalQuestions += 1

            #Here we concat the top 5 sentences for each question and process the stop words etc
            if len(result) > 0:
                bestSentenceText = article['sentences'][result[0][0]]  ############
                if len(result) > 1:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[1][0]] #######
                if len(result) > 2:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[2][0]] #######
                if len(result) > 3:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[3][0]] #######
                if len(result) > 4:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[4][0]] #######
                bestSentenceText = ''.join(ch for ch in bestSentenceText if ch not in PunctuationExclude)
                bestSentenceText = bestSentenceText.replace(",", " ,")
                bestSentenceText = bestSentenceText.replace(".", " .").split()
                bestSentenceTokensNoStopWords = []
                for i in range(0,len(bestSentenceText)-1):
                    if i >0:
                        if bestSentenceText[i] not in stopwords or bestSentenceText[i][0].isupper():
                            bestSentenceTokensNoStopWords.append(bestSentenceText[i])

                allBestSentences.append(bestSentenceTokensNoStopWords)
                allBestSentencesText.append(bestSentenceText)
                best = result[0][0]
                bestSentence[articleNo,questionNo] = best
                if qa['answer'] in bestSentenceText:   ##cheking the quality of our reterival, i.e. if answer is present in the fetched sentence
                    correctSentence += 1
            else:
                allBestSentences.append([]) #to preserve question sequence
                allBestSentencesText.append(" ")

            allQuestionText.append(qa['question']) #saving questions too for later usage

    #printing out reterival accuracy, #not much used, but can guide about the theorotical accuracy limit on the final QA system
    print("The reterival accuracy on test set is", (correctSentence/float(totalQuestions)))

    #Now computing NER and other tags (like POS if needed)
    print("Computing NER start at:", ctime())
    NER_tagged = stanford_NER_tagger.tag_sents(allBestSentences)
    print("NER Time:", ctime())
    print("NER Tagging Done, Now doing POS tagging")
    POS_taggedAnswers=[]
    # POS_taggedAnswers = stanford_POS_tagger.tag_sents(allBestSentencesText) ####Maybe needed for the 3rd answer ranking rule
    print("POS answer tagging Done")
    print("POS answer Time:", ctime())
    POS_taggedQuestions= []
    # POS_taggedQuestions = stanford_POS_tagger.tag_sents(allQuestionText) ####Maybe needed for the 3rd answer ranking rule
    print("POS question tagging Done")
    print("POS question Time:", ctime())


    #saving the computed NER tags and sentences
    f = open(fname, 'wb')  # 'wb' instead 'w' for binary file
    pickle.dump({'NER_tagged':NER_tagged, 'POS_taggedAnswers': POS_taggedAnswers, 'POS_taggedQuestions':POS_taggedQuestions,'allBestSentencesText':allBestSentencesText}, f, -1)  # -1 specifies highest binary protocol
    f.close()
    print("NER Saved")


else: #NER tagged found
    f = open(fname, 'rb')  # 'rb' for reading binary file
    #Loading saved variables
    allVars = pickle.load(f)
    NER_tagged = allVars['NER_tagged']
    POS_taggedAnswers = allVars['POS_taggedAnswers']
    POS_taggedQuestions = allVars['POS_taggedQuestions']
    allBestSentencesText = allVars['allBestSentencesText']
    f.close()
    print("All saved variables loaded")



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
}


#this list can used to check if a word is openClass or not, if we have POS tag
openClassTags = {
    'JJ',
    'JJR',
    'JJS',
    'NN',
    'NNP',
    'NNPS',
    'NNS',
    'RB',
    'RBR',
    'RBS',
    'UH',
    'VB',
    'VBD',
    'VBG',
    'VBN',
    'VBP',
    'VBZ'
}




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

        # Who
        # developed
        # the
        # ductile
        # form
        # of
        # tungsten?
#Trying to add NUMBER entity and removing ORGANIZATION
initial = 0
for answerSent in NER_tagged:
    for i in range (0,len(answerSent)-1):
        # tagging all other entities i.e. starts with capital and not tagged by NER
        if (answerSent[i][1] == 'O' and i > 0 and len(answerSent[i][0]) > 0 and answerSent[i][0][0].isupper()  and i > 0  and answerSent[i-1][0][0] != '.'):
            answerSent[i] = (answerSent[i][0], u'OTHER')
        # print(token)
        # Dis-regarding ORGINIZATION tag
        if answerSent[i][1] == "ORGANIZATION":
            answerSent[i] = (answerSent[i][0], u'OTHER')
            # print("****", answerSent[i][1])
        if is_number(answerSent[i][0]):
            answerSent[i] = (answerSent[i][0], u'NUMBER')
        if (i>0 and answerSent[i][0] != "," and  answerSent[i][0][0] == "," and is_number(answerSent[i][0][1:]) and answerSent[i-1][1] == 'NUMBER'):
            answerSent[i - 1] = (answerSent[i-1][0]+answerSent[i][0], u'NUMBER')



# These lists help to classify the question type, we just check if these words are in question
organizationList = {
    'company',
    'organization',
    'entity'
    # 'school',
    # 'college',
    # 'university',
    # 'team'

}
locationList = {
    'where',
    'location'
    # 'city',
    # 'country',
    # 'location',
    # 'continent',
    # 'state',
    # 'area',
    # 'river',
    # 'pond',
    # 'fall',
    # 'desert',
    # 'venue'

}
personList = {
    'who',
    'whom',
    'person',
    'scientist',
    'artist',
    'musician'
}

numberList = {
    'how many',
    'number',
    'count',
    'percent',
    'percentage',
    'when',
    'date',
    'year',
    'month',
    'day',
    'week'

}

#This is a helper function that check if a word in question appears in lists above
def checkWordInQuestion(question,wordList):
    for x in wordList:
        if x in question.lower():
            return True
    return False


## NOT NEEDED ANYMORE
# #Concatinating adjacent same tag entities
# for answerSent in NER_tagged:
#     CompactTagged = []
#     lastTag = None
#     for i in range(0,len(answerSent)-1):
#         for j in range (i+1,len(answerSent)-1):
#             if answerSent[i][1] == answerSent[j][1]:

# Build a simple question classifier based on type of wh word in question:
def classifyQuestion(question):
    if  checkWordInQuestion(question,organizationList):
        return "OTHER"
    elif checkWordInQuestion(question,locationList) :
        return "LOCATION"
    elif checkWordInQuestion(question,personList):
        return "PERSON"
    elif checkWordInQuestion(question,numberList):
        return "NUMBER"
    else:
        return "OTHER"



# These counters, count the statistics on the train set
correct = 0
possCorrect = 0
wrongNumber = 0
totalans = 0
multiAnswer = 0
x=0
i = -1 #index of our NER_TAGGED list (i.e. questions)
for article in data:
    for question in article['qa']:  #For all questions in the article, but notice that we add all questions to the same list in the end, indexed by i
        i+=1
        taggedBestAnswerSent = NER_tagged[i]
        answerSentText = u" ".join(allBestSentencesText[i])   # as we have sentence in the form of token lists so we join it into single string
        questionType = classifyQuestion(question['question']) #guess the question type, based on words in the question text
        answerList = []
        t=0
        #trying to find questionType entity in answer

        #We find all the entities of question type in the answer text, i.e. the first rule of answer filtering
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
            if guessedAnswerText != "":
                answerList.append(guessedAnswerText) #collect all the candidate answers seen



        #we didnt find any matching entity type so we will give OTHER entity as answer
        if (len(answerList) < 1):
            x+=1
            questionType = "OTHER"
            t=0
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
                answerList.append(guessedAnswerText)  # collect all the candidate answers seen
            # print(question['question'])
            # print(taggedBestAnswerSent)
            # print(answerList)
            # print("-----" + question['answer'])


        guessedAnswerText = ""
        filteredAnswers = []




        #Filter as per second rule, that if candidate answer is fully included in the question we disregard it
        for ans in answerList:
            if ans not in question['question']:
                filteredAnswers.append(ans)

        #Trying to apply 3rd rule, but currently not stable all is commented and we select the first candidate ansewer
        if (len(filteredAnswers) > 0):
            totalans += len(filteredAnswers)
            multiAnswer += 1
            guessedAnswerText = filteredAnswers[0]

            # openClassInQuestion = []
            # openClassInAnswer = []
            #
            #
            # posTaggedQuestion = POS_taggedQuestions[i]
            # posTaggedAnswer = POS_taggedAnswers[i]
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
            # ## Find same in both
            # setSame = set(openClassInAnswer) & set(openClassInQuestion)
            #
            # distancesFromOpenClass={}
            #
            # print(filteredAnswers)
            # for possibleAnswer in filteredAnswers:
            #     for questionWord in setSame:
            #         # print(answerSentText)
            #
            #
            #         dist = answerSentText.find(possibleAnswer[1:])
            #         if dist < 0:
            #             dist = 10000000000000
            #         distancesFromOpenClass[possibleAnswer[1:],questionWord] = dist
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
            # minDist = 1000000000000
            # minA = ""
            # for A in distanceAnswer:
            #     if distanceAnswer[A] < minDist:
            #         minDist = distanceAnswer[A]
            #         minA = A
            #
            # guessedAnswerText  = minA






            #This loop is just for train statistics that if the answer was reterived but not the first candidate answer, so we can improve our selection in future builds
            for ansC in filteredAnswers:
                if ansC[1:] == question['answer'] and filteredAnswers[0]!= ansC:
                    # print(question["question"],"::",NER_tagged[i])
                    # print(questionType, "::", filteredAnswers, question["answer"])
                    possCorrect+=1
                    break
        else:
            if (len(answerList) > 0):
                guessedAnswerText = answerList[0]
            else:
                guessedAnswerText = ""


        #Cleaning the answers a little bit, got these via errror analysis
        PunctuationExclude = set(string.punctuation)
        PunctuationExclude.remove(',')
        PunctuationExclude.remove('-')
        PunctuationExclude.remove('.')
        PunctuationExclude.remove('\'')
        PunctuationExclude.remove('%')
        guessedAnswerText = ''.join(ch for ch in guessedAnswerText if ch not in PunctuationExclude)  ######
        if guessedAnswerText != "":
            guessedAnswerText = guessedAnswerText[1:]  # remove the first space
            # print(guessedAnswerText)


        if(questionType == 'NUMBER' and '.' in guessedAnswerText):
            guessedAnswerText = guessedAnswerText.replace(" ", "")
        if (questionType == 'NUMBER' and '%' in guessedAnswerText):
            guessedAnswerText = guessedAnswerText.replace(" ", "")
        if (questionType == 'NUMBER' and ('what year' in question["question"].lower() or 'which year' in question["question"].lower() )):
            for ans in filteredAnswers:
                try:
                    guessedAnswerText =  str(parse(ans, fuzzy=True).year)
                    break
                except ValueError:
                    guessedAnswerText =guessedAnswerText
                    continue





        #here we finalize the answer for this question and check it for stats
        if guessedAnswerText == question['answer']:
            correct +=1

        elif questionType == 'NUMBER':
            wrongNumber += 1
            print(i,": ",question['question'])
            print(taggedBestAnswerSent)
            print(filteredAnswers)
            print(guessedAnswerText)
            print("-----" + question['answer'])

print("wrong in selected cat",wrongNumber)
print("total",i)
print("correct",correct)
print("correct in multi ans",possCorrect)
print("avg multi ans len", totalans/float(multiAnswer))
print("Accuracy ",correct/float(i))

print("All Answer Computation Time:", ctime())
print(initial)