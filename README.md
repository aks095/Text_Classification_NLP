# Text Classification with NLP: TF-IDF vs. Word2Vec vs. BERT

[Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

[Repo Link](https://github.com/Legolas2215/News_Category_Classification.git)

This repository provides code and resources for exploring three popular approaches for text classification using natural language processing (NLP) techniques: TF-IDF, Word2Vec, and BERT.
Text classification is a fundamental task in NLP, with applications ranging from sentiment analysis and topic categorization to spam detection and intent recognition. The goal is to automatically assign predefined categories or labels to text documents based on their content.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Approaches](#approaches)
  - [TF-IDF](#tf-idf)
  - [Word2Vec](#word2vec)
  - [BERT](#bert)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

This repository provides a comparison of three different techniques for text classification: TF-IDF, Word2Vec, and BERT. 

## Setup
- Python 3.x
- Jupyter Notebook
- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- Gensim
- NLTK
- Hugging Face Transformers
  
## Approaches

### TF-IDF

The TF-IDF (Term Frequency-Inverse Document Frequency) approach represents documents as vectors by considering the importance of each term in a document relative to a collection of documents. It is a simple yet effective technique for text classification, where each document is transformed into a sparse vector of TF-IDF weights. The article explains the underlying calculations and demonstrates how to implement TF-IDF for text classification tasks.

### Word2Vec

The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. As the name implies, word2vec represents each distinct word with a particular list of numbers called a vector. The vectors are chosen carefully such that they capture the semantic and syntactic qualities of words; as such, a simple mathematical function (cosine similarity) can indicate the level of semantic similarity between the words represented by those vectors.

### BERT

BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art language model that captures contextual information from both the left and right context of a word. It has revolutionized many NLP tasks, including text classification.

## Usage

To use the code provided in this repository, follow the instructions below:

1. Clone the repository:

2. Install the required dependencies as mentioned in the [Setup](#setup) section.

3. Explore the Jupyter Notebooks in the repository to understand the implementation of each approach. The notebooks provide step-by-step instructions, code snippets, and explanations to guide you through the process.

4. Customize the code or experiment with your own datasets to further explore the text classification techniques discussed in the article.

## Results

The code compares the performance metrics, such as accuracy, precision, recall, and F1 score, achieved by TF-IDF, Word2Vec, and BERT. The results highlight the strengths and limitations of each approach and provide insights into their effectiveness for different text classification tasks.



