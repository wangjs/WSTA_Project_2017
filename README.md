# Introduction
In this project, we implemented a Question Answering System. The goal of the system was given a
Wikipedia article and a list of question based on the article, answer as many of them as possible. As per
the project specification document as Baseline QA system was developed which yielded 12.5%
accuracy. Then we implemented 3 enhancements, mainly: a better question type classifier, better
sentence retrieval engine and a probabilistic QA system based on semantic cues. These enhancements
enabled us to achieve maximum accuracy of 13.2%. In this report, we will outline the challenges we
faced and the solutions we implemented



All files reffered below are present in WSTA_Submit Directory.

Final Project report is also available in Submit Directory.

# Team:
Muhammad Umer Altaf (778566),
Lufan Zhang (827495),
Channing Che (823488)

---Kaggle Team Name: “UniMelb”---

# Instructions to run this project:
Please ensure you have following dataset files in the same directory as these scripts:
QA_dev.json
QA_train.json
QA_test.json



# For Base QA:
- Follow following instructions:
- Run: BaseQA.py

# For Enhanced QA (Enhancement 1 - Better Question Classifier):
- Follow following instructions:
- Run: GenerateQuestionLabels.py (This will generate the dataset for the classifier)
- Run: BuildQuestionClassifier.py (This will fit a model on the data set generated in previous step )
- Run: EnhancedQA.py (Depending on RunOn switch in this file Accuracy will be calculated on dev set OR .csv will be generated for testset)

-- Muhammad Umer Altaf
(Other member's contribution is present in directories named by their first name)
For work division, please refer to contribution report.
