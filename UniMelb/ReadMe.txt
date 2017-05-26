Team:
Muhammad Umer Altaf (778566),
Lufan Zhang (827495),
Channing Che (823488)

---Kaggle Team Name: “UniMelb”---

Instructions to run this project:
Please ensure you have following dataset files in the same directory as these scripts:
QA_dev.json
QA_train.json
QA_test.json



For Base QA:
# Follow following instructions:
# Run: BaseQA.py

For Enhanced QA (Enhancement 1 - Better Question Classifier):
# Follow following instructions:
# Run: GenerateQuestionLabels.py (This will generate the dataset for the classifier)
# Run: BuildQuestionClassifier.py (This will fit a model on the data set generated in previous step )
# Run: EnhancedQA.py (Depending on RunOn switch in this file Accuracy will be calculated on dev set OR .csv will be generated for testset)

-- Muhammad Umer Altaf
(Other member's contribution is present in directories named by their first name)
For work division, please refer to contribution report.
