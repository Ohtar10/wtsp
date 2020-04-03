# Where To Sell Products
#### Author: Luis Eduardo Ferro Diez <a href="mailto:luisedof10@gmail.com">luisedof10@gmail.com</a>

This repository contains all my work for my MsC in Computer Science project.

Where To Sell Products (wtsp) is a project in which I try to solve this very same question. The idea is to characterize geographic areas in terms of its relationship with a selected set of products. The relationship is built from natural language, i.e., I gathered geotagged text data from Twitter and product reviews from Amazon. For the former I generated spatial clusters from which all the tweets are aggregated to form a single cluster-corpus. For the latter I trained a convolutional neural network to classify product categories given several review text. Finally each cluster-corpus is submitted to the classifier to emit a relevance score for some categories. The result is displayed on a map of a study area, e.g., a city, in which the clusters are shown with their corresponding relationship score with certain products and services.

## Demo of wtsp in Los Angeles
![](./media/wtsp-demo.gif)

## Repository Structure
* [Data Preparation (dataprep)](dataprep/)
* [Notebooks](notebooks/)
* [Documentation](documentation/)
* [Runtime environments (env)](env/)

## System Requirements
* Java 1.8.x
* Apache Spark >= 2.3
* Conda >= 4.7.x
* Python 3.6.x
* CUDA 9.1
* Tensorflow 1.12.0
* Keras 2.2.4

## Workflow
### 1. Gather the data
* Data from twitter can be obtained from https://archive.org <a href="https://archive.org/details/twitterstream&tab=collection">here</a>.
* Data from Amazon product reviews can be obtained <a href="http://jmcauley.ucsd.edu/data/amazon/">here</a>.

### 2. Execute the data preparation pipelines
Both twitter and product review data needs to be pre-processed, for this there are two spark projects under [dataprep](dataprep/).
#### Tweets Transformer
This job takes the twitter data, filters out the tweets that are not geotagged and sinks the result as parquet.
#### Amazon Product Reviews Transformer
This job takes the raw amazon product reviews and product metadata and converts them into 'documents' where each document is a category and either a review text or a product description.

### 3. Train the product classifier
Install the [cli](cli/) and follow the instructions to create the embeddings and train the classifier with the transformed product documents.

### 4. Predict the geographic area categories
Use the [cli](cli/) to predict the detect and classify the geographic
areas.

## License

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

See the LICENSE_ file in the root of this project for license details.
