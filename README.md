# Distinguishing Comparison in Sentences
## Short description
This repository contains the code of the final project in the Machine Learning 2020 Course "Distinguishing Comparison in Sentences". The work's aim is to reproduce the results achieved by the best models from the work ["Categorizing Comparative Sentences"](https://arxiv.org/abs/1809.06152) and improve them by using other machine learning models (including transformers).  
The general task is to perform categorization of sentences into three classes: ``BETTER``, ``WORSE``, or ``NONE`` where each sentence is expected to contain mentions of two objects under a comparison. For instance, consider the following sentence: 
``Python is better than Ruby for scientific programming``. In this sentence, Python is compared to Ruby with respect to the aspect "scientific programming". Our classifiers are expected to categorize it with the ``BETTER`` label as the first mentioned object is better than the second mentioned objeсt.
The models presented:
- BOW + XGBoost (from the paper)
- InferSent + XGBoost (from the paper)
- BERT classifier  

The code for reproducing the results was partially taken from the repo:  
https://github.com/uhh-lt/comparative  
In the experiments we rely on libraries:
- [InferSent](https://github.com/facebookresearch/InferSent)
- [AllenNLP](https://github.com/allenai/allennlp)
- [pytorch-pretrained-bert](https://github.com/maknotavailable/pytorch-pretrained-BERT)

## Brief results
The metric is F1-score for each of the classes and micro-averaged F1-score in general:
| Model               | BETTER | WORSE  | NONE   | AVG    |
| ------------------- |:------:|:------:|:------:|:------:|
| BOW + XGBoost       | 0.758  | 0.408  | 0.925  | 0.859  |
| InferSent + XGBoost | 0.759  | 0.431  | 0.921  | 0.858  |
| ELMo + LogReg       | 0.756  | 0.424  | 0.926  | 0.861  |
| BERT classifier     | 0.799  | 0.643  | 0.935  | 0.885  |

## Instructions to running the code
### To run reproduction of the results (BOW + XGBoost and InferSent + XGBoost) and ELMo + classifiers experiments
1. Clone the repo.
2. Download the file [``infersent.allnli.pickle``](https://drive.google.com/open?id=1G9qEKCmRo3pegwlQEHbe0pZxFRSFop3n) from the [google drive](https://drive.google.com/open?id=1G9qEKCmRo3pegwlQEHbe0pZxFRSFop3n) and put in the folder ``infersent/``.
3. Download Glove embeddings (glove.840B.300d.zip) from the [official site](https://nlp.stanford.edu/projects/glove/) or from the [google drive](https://drive.google.com/open?id=1JVI2jk1xBmgt8iMy7U3HYDUesGXgvi73) and unzip them in the folder ``infersent/``.
4. Install dependecies. The dependencies are listed in the ``requirements.txt`` file and you can install them using ``pip3 install -r requirements.txt``. IMPORTANT: one of the requirements is ``torch`` and installation may fail, so you should install it using instructions from the [official site](https://pytorch.org/get-started/locally/).
5. Open the notebook ``Final_project.ipynb`` and run it from the start.
### To run the BERT model
BERT demands higher computing powers, so it was run on colab from the start. To run it just open the notebook [``BERT_classifier.ipynb``](https://colab.research.google.com/drive/1yLxIhWBUiq2jHk-5npeB_EXhJDVOLYZg) in colab and follow the cells. Extra code will be loaded from google drive and the library ``pytorch-pretrained-bert`` will be installed by the way.
