# Artificial Intelligence - Artificial Neural Networks

## Authors
- Mateusz Jakubczak
- Krzysztof Olipra
- Karol Oleszek

## Objective

Will client open a deposit? - binary classification based on telemarketing data.

## Structure

Project structure:

- bank.csv - UCI dataset
- ssn.py - artificial neural network implementation
- compare_methods.py - comparative study of alternative methods
- compare_methods_report.txt - detailed report from comparative study

## Dataset

Data was downloaded from [open machine learning dataset repository](https://archive.ics.uci.edu/ml/datasets/Bank%2BMarketing).

## Problem description

Dataset was gathered from marketing activities of Portugeese commercial bank.

Client cold calls was main marketing activity.

Classification objective:

- Predict whether client will open a deposit after telemarketing call

## Client features

- age
- job
- is married
- education
- has defaulted on loan
- has mortgage
- has loans


## Previous calls features

- has home phone
- time from last contact
- weekday of last contact
- call time

## Other features

- number of previous client calls
- were previous calls successful

## Macroeconomic features

- quarterly unemployment variance
- monthly CPI
- Consumer Confidence Index
- euribor 3 rate
- employees in economy


## Related work

S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimaraes, Portugal, October, 2011. EUROSIS. [bank.zip]

## Other methods - comparative study

Other approaches:

- Decision trees
- Naive Bayes
- K-nearest neighbors
- Support Vector Machines


# Porównanie wyników - inne metody

|Method|Accuracy|
|---|---|
|SSN|64%|
|Decision tree|68%|
|Naive Bayes|68%|
|KNN|60%|
|SVC|58%|
