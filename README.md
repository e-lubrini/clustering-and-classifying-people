## UE 803: Data Science
# Project: Clustering and Classifying People based on Text and KB information

## Installation

1. cd to the directory where ```requirements.txt``` is located;

2. activate your virtualenv;
3. run: `pip install -r requirements.txt` in your shell.

## Usage

In order to obtain all the results, it is necessary to run 4 following commands:

```python3 scripts/corpus_extraction.py --n_people 30 --n_sentences 10 --verbose```

```python3 scripts/preprocessing.py --input data/data.csv --output data/preprocessed_data.csv --verbose```

```python3 scripts/clustering.py --input data/preprocessed_data.csv --verbose```

```python3 scripts/classification.py --input data/preprocessed_data.csv --verbose```

**```corpus_extraction.py```** script extracts the wikidata articles ids using the sparqle query on wikidata data. These articles should contain one of the following keywords: *architect, mathematician, painter, politician, singer or writer*. This script automatically creates an additional file ```data/keywords_ids.json``` where all the ids related to the given keywords are stored (if provided, the script will use this file instead of parsing wikidata as it takes time to run it from scratch). After retrieveing all the ids, the script extracts *M* (```n_people``` param) articles from each category and stores their title, description and *N* (```n_sentences``` param) sentences from the main text. The result is then stored into the ```data/data.csv``` file. 

**```preprocessing.py```** script takes as an input the path to the csv file (--input param, default value is ```data/data.csv```) with the data and processes it using the following transformations (these are the default values in the ```Preprocess``` class which may be changed there): 

- Lower casing
- Removing the stop words
- Removing the punctuation
- Removing all the letters that are not from the English alphabet
- Lemmatizing

the preprocessed file is then stored into the default (```data/preprocessed_data.csv```) file or another one defined in the ```--output``` param. 

**```clustering.py```** script takes as an input the path to the preprocessed csv file which will be used for training the KMeans algorithm (```--input param```). All the data is preprocessed using three methods: tf-idf, token presence and token frequency. Two ways of clustering are applied: clustering into 2 classes and claustering into 6 classes. The results of the algorithm with the default parameters are then stored into the ```data/Clustering results.csv``` (a comparison table) and ```data/Clustering visualization.png``` (a comparison plot) files. 

**```classification.py```** script  takes as an input the path to the preprocessed csv file which will be used for training the Percetron algorithm. All the data is preprocessed using Tf-Idf Vectorizer. The results of the algorithm with th default parameters are stored into the data folder. The files are:

- ```Confusion matrix 6 classes.png``` with the confusion matrix for the 6 class classification

- ```Confusion matrix 2 classes.png``` with the confusion matrix for the 2 class classification

- ```Scores 6 classes.csv``` with the Recall, Precision and F-score computed for the 6 class classification

- ```Scores 2 classes.csv``` with the Recall, Precision and F-score computed for the 2 class classification

- ```Accuracy visualization.png``` with the comparison of the per class acurracy for the 2 and 6 class classification algorithms. The metric is defined by the following formula: (TP+TN)/(TP+TN+FP+FN), where TP are correct cases for the target category, TN are correct cases when an example was not classified as a target category. 

   

All the scripts have the ```--verbose``` param which activates printing out the main steps of the algortihms during their execution.

## Results

These are the results obtained on the data stored in the ```data``` directory:

**Clusterization**:

![Clustering visualization](/Users/anna/Documents/GitHub/clustering-and-classifying-people/data/Clustering visualization.png)



| **2 clust., tf-idf**       | **6 clust., tf-idf** | **2 clust., tokens** | **6 clust., tokens** | **2 clust., token freq** | **6 clust., token freq** |
| -------------------------- | -------------------- | -------------------- | -------------------- | ------------------------ | ------------------------ |
| **0.00422224586443926**    | 0.07869913816783629  | 0.005577943863212365 | 0.06770759388899555  | 0.012803360133562881     | 0.1727469447581936       |
| **0.005149298161543911**   | 0.08312414696925129  | 0.1124266459657679   | 0.13524027725716187  | 0.01931815408279083      | 0.17548881502174843      |
| **0.004639919058601989**   | 0.08085114230470347  | 0.010628561496381642 | 0.09023788934813323  | 0.015400101139173863     | 0.17410708569045097      |
| **0.00019394127458205655** | 0.022563581695685914 | 0.0                  | 0.032766745688207234 | 0.006836040880631517     | 0.08332501447481383      |
| **0.0031097721509188406**  | 0.006967920192568018 | 0.5711080500748351   | 0.017397227636989858 | 0.1703049285016345       | 0.05498191279222395      |

**Classification**:

**2 class classification:**

![Confusion matrix 2 classes](/Users/anna/Documents/GitHub/clustering-and-classifying-people/data/Confusion matrix 2 classes.png)

|                  | **precision** | **recall** | **f1-score** | **support** |
| ---------------- | ------------- | ---------- | ------------ | ----------- |
| **A**            | 0.11          | 0.6        | 0.19         | 5.0         |
| **Z**            | 0.93          | 0.51       | 0.66         | 49.0        |
| **accuracy**     | 0.52          | 0.52       | 0.52         | 0.52        |
| **macro avg**    | 0.52          | 0.56       | 0.42         | 54.0        |
| **weighted avg** | 0.85          | 0.52       | 0.61         | 54.0        |

**6 class classification:**

![Confusion matrix 6 classes](/Users/anna/Documents/GitHub/clustering-and-classifying-people/data/Confusion matrix 6 classes.png)

|                   | **precision** | **recall** | **f1-score** | **support** |
| ----------------- | ------------- | ---------- | ------------ | ----------- |
| **architect**     | 0.33          | 0.23       | 0.27         | 13.0        |
| **mathematician** | 0.22          | 0.25       | 0.24         | 8.0         |
| **painter**       | 0.22          | 1.0        | 0.36         | 2.0         |
| **politician**    | 0.11          | 0.14       | 0.12         | 7.0         |
| **singer**        | 0.56          | 0.33       | 0.42         | 15.0        |
| **writer**        | 0.44          | 0.44       | 0.44         | 9.0         |
| **accuracy**      | 0.31          | 0.31       | 0.31         | 0.31        |
| **macro avg**     | 0.31          | 0.4        | 0.31         | 54.0        |
| **weighted avg**  | 0.36          | 0.31       | 0.32         | 54.0        |

**Comparison**:

![Accuracy visualization](/Users/anna/Documents/GitHub/clustering-and-classifying-people/data/Accuracy visualization.png)



