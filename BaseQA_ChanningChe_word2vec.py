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

from operator import itemgetter
from gensim import corpora, models, similarities

runOn = "DEV"


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
#os.environ["STANFORD_MODELS"] = "/chechi/Documents/StanfordNER"
stanford_NER_tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
stanford_POS_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
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
        data = json.load(data_file)[51:101]



# getting the list of englist stopwords
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.remove('the') ## After the error analysis of the results I realised that many answers have these words i.e. The President
stopwords.remove('of') ## So will not exclude these


stopwordsAll = set(nltk.corpus.stopwords.words('english'))


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
    allSecondBestSentencesText = []
    allSecondBestSentences = []
    allBestSentencesText = []
    allQuestionText = []
    
    articleNo = -1
    for article in data:
        articleNo += 1
        print("Computing Article: ",articleNo+1,'/',len(data))
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        allSentenceList = []
        for sentence in article['sentences']:
            sentenceText = ''.join(ch for ch in sentence if ch not in PunctuationExclude)
            sentenceText=sentenceText.replace(",", " ,")
            sentenceText=sentenceText.replace(".", " .")
            sentenceList = [word for word in sentenceText.lower().split() if word not in stopwords]
            allSentenceList.append(sentenceList)
        dictionary = corpora.Dictionary(allSentenceList)
        corpus = [dictionary.doc2bow(sentenceList) for sentenceList in allSentenceList]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=500)
        index = similarities.MatrixSimilarity(lsi[corpus])
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        '''
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
        '''
        #now for each question we reterive a list of most relevent sentences
        questionNo = -1
        for qa in article['qa']:
            questionNo += 1
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            question = qa['question']
            questionText = ''.join(ch for ch in question if ch not in PunctuationExclude)
            questionText = questionText.replace(",", " ,")
            questionText = questionText.replace(".", " .")
            questionList = [word for word in questionText.lower().split() if word not in stopwords]
            que_bow = dictionary.doc2bow(questionList)#questionText.lower().split())
            que_lsi = lsi[que_bow]
            sims = index[que_lsi]
            sims = list(enumerate(sims))
            sims = dict(sims)
            sims = sorted(sims.items(),key=itemgetter(1),reverse=True)
            result = sims
            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            '''
            query = ""
            questionText = qa['question']
            questionText = ''.join(ch for ch in questionText if ch not in PunctuationExclude) ######
            questionText = questionText.replace(",", " ,")
            questionText = questionText.replace(".", " .")
            for token in nltk.word_tokenize(questionText):
                if token not in stopwords:  # 'in' and 'not in' operations are much faster over sets that lists
                    query = query + ' ' + token
            result = query_vsm([stemmer.stem(term.lower()) for term in query.split()], vsm_inverted_index)
            '''

            totalQuestions += 1

            #Here we concat the top 5 sentences for each question and process the stop words etc
            if len(result) > 0:
                bestSentenceText = article['sentences'][result[0][0]]  ############
                if len(result) > 1:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[1][0]] #######
                bestSentenceText = ''.join(ch for ch in bestSentenceText if ch not in PunctuationExclude)
                bestSentenceText = bestSentenceText.replace(",", " ,")
                bestSentenceText = bestSentenceText.replace(".", " .").split()
                bestSentenceTokensNoStopWords = []
                for i in range(0,len(bestSentenceText)-1):
                    if i >0:
                        if bestSentenceText[i] not in stopwords or bestSentenceText[i][0].isupper():
                            bestSentenceTokensNoStopWords.append(bestSentenceText[i])
                    else:
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




            if len(result) > 2:
                bestSentenceText = article['sentences'][result[2][0]]  ############
                if len(result) > 3:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[3][0]] #######
                if len(result) > 4:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[4][0]] #######
                if len(result) > 5:
                    bestSentenceText = bestSentenceText + " " + article['sentences'][result[5][0]]  #######
                bestSentenceText = ''.join(ch for ch in bestSentenceText if ch not in PunctuationExclude)
                bestSentenceText = bestSentenceText.replace(",", " ,")
                bestSentenceText = bestSentenceText.replace(".", " .").split()
                bestSentenceTokensNoStopWords = []
                for i in range(0,len(bestSentenceText)-1):
                    if i >0:
                        if bestSentenceText[i] not in stopwords or bestSentenceText[i][0].isupper():
                            bestSentenceTokensNoStopWords.append(bestSentenceText[i])
                    else:
                        bestSentenceTokensNoStopWords.append(bestSentenceText[i])

                allSecondBestSentences.append(bestSentenceTokensNoStopWords)
                allSecondBestSentencesText.append(bestSentenceText)
                best = result[0][0]
                bestSentence[articleNo,questionNo] = best
                if qa['answer'] in bestSentenceText:   ##cheking the quality of our reterival, i.e. if answer is present in the fetched sentence
                    correctSentence += 1
            else:
                allSecondBestSentences.append([]) #to preserve question sequence
                allSecondBestSentencesText.append(" ")


            # if len(result) > 2:
            #     bestSentenceText = bestSentenceText + " " + article['sentences'][result[2][0]] #######
            # if len(result) > 3:
            #     bestSentenceText = bestSentenceText + " " + article['sentences'][result[3][0]] #######
            # if len(result) > 4:
            #     bestSentenceText = bestSentenceText + " " + article['sentences'][result[4][0]] #######




            allQuestionText.append(qa['question']) #saving questions too for later usage

    #printing out reterival accuracy, #not much used, but can guide about the theorotical accuracy limit on the final QA system
    print("The reterival accuracy on test set is", (correctSentence/float(totalQuestions)))

    #Now computing NER and other tags (like POS if needed)
    print("Computing NER start at:", ctime())

    NER_tagged = stanford_NER_tagger.tag_sents(allBestSentences)
    print("NER Time 1:", ctime())
    NER_tagged2 = stanford_NER_tagger.tag_sents(allSecondBestSentences)

    print("NER Time 2:", ctime())
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
    pickle.dump({'NER_tagged':NER_tagged,
                 'POS_taggedAnswers': POS_taggedAnswers,
                 'POS_taggedQuestions':POS_taggedQuestions,
                 'allBestSentencesText':allBestSentencesText,
                 'NER_tagged2' : NER_tagged2,
                 'allSecondBestSentencesText' : allSecondBestSentencesText


                 }, f, -1)  # -1 specifies highest binary protocol
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
    NER_tagged2 = allVars['NER_tagged2']
    allSecondBestSentencesText =  allVars['allSecondBestSentencesText']
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
    # 'year'
    # 'years',
    # 'day',
    # 'days',
    # 'month',
    # 'months'
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
    # 'desert',
    # 'venue'

}
personList = {
    'who',
    'whom',
    'person'
    # 'scientist',
    # 'artist',
    # 'musician',
    # 'inventor',
    # 'son',
    # 'father',
    # 'daughter',
    # 'sister',
    # 'brother'
}

numberList = {
    'how many',
    'how much'
    'number',
    'count',
    'percent',
    'percentage',
    'when',
    'date',
    'year',
    'month',
    'day',
    'week',
    'version',
    'how',
    'ammount',
    'rate'

}

def isSentCase(sent):
    for x in sent.split():
        if not x[0].isupper():
          return False
    return True

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
blank=0
x=0


def extractAnswer(questionType,taggedBestAnswerSent,answerSentText,guessOTHERtype,apply3rdRule):
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

    if(guessOTHERtype):
        allQTypesList = ["NUMBER","PERSON","LOCATION","OTHER"]
    # we didnt find any matching entity type so we will give OTHER entity as answer
        if (len(answerList) < 1):
            questionType = "OTHER"
            t = 0
            for t in range(0, len(taggedBestAnswerSent) - 1):
                guessedAnswerText = ""
                if taggedBestAnswerSent[t][1] in allQTypesList :
                    for l in range(t, len(taggedBestAnswerSent) - 1):
                        if taggedBestAnswerSent[l][1] ==  taggedBestAnswerSent[t][1] :
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
        minDist = 9999999999999999999
        minAns = ""
        for ans in answerDist:
            if answerDist[ans] < minDist:
                minAns = ans
                minDist = answerDist[ans]

        guessedAnswerText = minAns

    # Cleaning the answers a little bit, got these via errror analysis
    PunctuationExclude = set(string.punctuation)
    PunctuationExclude.remove(',')
    PunctuationExclude.remove('-')
    PunctuationExclude.remove('.')
    PunctuationExclude.remove('\'')
    PunctuationExclude.remove('%')
    guessedAnswerText = ''.join(ch for ch in guessedAnswerText if ch not in PunctuationExclude)  ######

    if (questionType == 'NUMBER' and '.' in guessedAnswerText):
        guessedAnswerText = guessedAnswerText.replace(" ", "")
    if (questionType == 'NUMBER' and '%' in guessedAnswerText):
        guessedAnswerText = guessedAnswerText.replace(" ", "")
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
    return guessedAnswerText,filteredAnswers



i = -1 #index of our NER_TAGGED list (i.e. questions)
for article in data:
    for question in article['qa']:  #For all questions in the article, but notice that we add all questions to the same list in the end, indexed by i
        i+=1
        taggedBestAnswerSent = NER_tagged[i]
        taggedSecondBestAnswer = NER_tagged2[i]

        answerSentText = u" ".join(allBestSentencesText[i])   # as we have sentence in the form of token lists so we join it into single string
        secondAnswerSentText = u" ".join(allSecondBestSentencesText[i])   # as we have sentence in the form of token lists so we join it into single string
        questionType = classifyQuestion(question['question']) #guess the question type, based on words in the question text

        # if(i==8866):
        #     print(taggedBestAnswerSent)
        #     print(answerSentText)
        guessedAnswerText,filteredAnswers = extractAnswer(questionType,taggedBestAnswerSent,answerSentText,False,True)


        #our top most 2 candidate sentences did not give any answers so now we search in 3rd 4th and 5th sentence combined
        if(guessedAnswerText =="" or guessedAnswerText == " " or guessedAnswerText.lower() in stopwordsAll):
            guessedAnswerText,filteredAnswers = extractAnswer(questionType,taggedSecondBestAnswer,secondAnswerSentText,False,True)

        if (guessedAnswerText == "" or guessedAnswerText == " " or guessedAnswerText.lower() in stopwordsAll):
            guessedAnswerText,filteredAnswers = extractAnswer(questionType, taggedBestAnswerSent, answerSentText, True, False)

        if (guessedAnswerText == "" or guessedAnswerText == " " or guessedAnswerText.lower() in stopwordsAll):
            guessedAnswerText,filteredAnswers = extractAnswer(questionType, taggedSecondBestAnswer, secondAnswerSentText, True, False)
        if (guessedAnswerText == "" or guessedAnswerText == " " or guessedAnswerText.lower() in stopwordsAll):
            blank+=1

#here we finalize the answer for this question and check it for stats
        if guessedAnswerText == question['answer']:
            correct +=1

        elif questionType == 'OTHER':
            wrongNumber += 1
            print(i, ": ", question['question'],question['answer'],"-",guessedAnswerText)
            print(taggedBestAnswerSent)
            print(" ")
            # print(filteredAnswers)
            # print(guessedAnswerText)
            # print("-----" + question['answer'])

print("wrong in selected cat",wrongNumber)
print("total",i)
print("correct",correct)
print("correct in multi ans",possCorrect)
# print("avg multi ans len", totalans/float(multiAnswer))
print("Accuracy ",correct/float(i))

print("All Answer Computation Time:", ctime())
print(blank)