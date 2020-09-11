# Where To Sell Products
#### Author: Luis Eduardo Ferro Diez <a href="mailto:luisedof10@gmail.com">luisedof10@gmail.com</a>

This repository contains all my work for my MsC in Computer Science project.

Where To Sell Products (wtsp) is a project in which I try to solve this very same question. The idea is to characterize geographic areas in terms of its relationship with a selected set of products. The relationship is derived from geotagged texts, i.e., I gathered geotagged text data from Twitter and product reviews from Amazon. For the former I generated spatial clusters from which all the tweets are aggregated to form a single cluster-corpus. For the latter I trained a convolutional neural network to classify product categories given several review texts. Finally each cluster-corpus is submitted to the classifier to emit a relevance score for some categories. The result is displayed on a map of an area of interest, e.g., a city, in which the clusters are shown with their corresponding relationship score with certain products categories.

## Demo
### Los Angeles (2013-07)
![](./media/wtsp-demo-la.gif)

### Vancouver (2013-07)
![](./media/wtsp-demo-vancouver.gif)

### New York (2013-07)
![](./media/wtsp-demo-newyork.gif)

## Live Demo

| City       | Date     | 
| :------------- | :----------: |
|  Los Angeles | [June 2013](demo/where_to_sell_in/2013-06/place_name=Los%20Angeles/classified_clusters.html)   |
|  Los Angeles | [July 2013](demo/where_to_sell_in/2013-07/place_name=Los%20Angeles/classified_clusters.html)   |
|  Los Angeles | [August 2013](demo/where_to_sell_in/2013-08/place_name=Los%20Angeles/classified_clusters.html)   |
|  Los Angeles | [September 2013](demo/where_to_sell_in/2013-09/place_name=Los%20Angeles/classified_clusters.html)   |
|  New York | [June 2013](demo/where_to_sell_in/2013-06/place_name=New%20York/classified_clusters.html)   |
|  New York | [July 2013](demo/where_to_sell_in/2013-07/place_name=New%20York/classified_clusters.html)   |
|  New York | [August 2013](demo/where_to_sell_in/2013-08/place_name=New%20York/classified_clusters.html)   |
|  New York | [September 2013](demo/where_to_sell_in/2013-09/place_name=New%20York/classified_clusters.html)   |
|  Vancouver | [June 2013](demo/where_to_sell_in/2013-06/place_name=Greater%20Vancouver/classified_clusters.html)   |
|  Vancouver | [July 2013](demo/where_to_sell_in/2013-07/place_name=Greater%20Vancouver/classified_clusters.html)   |
|  Vancouver | [August 2013](demo/where_to_sell_in/2013-08/place_name=Greater%20Vancouver/classified_clusters.html)   |
|  Vancouver | [September 2013](demo/where_to_sell_in/2013-09/place_name=Greater%20Vancouver/classified_clusters.html)   |
|  Toronto | [June 2013](demo/where_to_sell_in/2013-06/place_name=Toronto/classified_clusters.html)   |
|  Toronto | [July 2013](demo/where_to_sell_in/2013-07/place_name=Toronto/classified_clusters.html)   |
|  Toronto | [August 2013](demo/where_to_sell_in/2013-08/place_name=Toronto/classified_clusters.html)   |
|  Toronto | [September 2013](demo/where_to_sell_in/2013-09/place_name=Toronto/classified_clusters.html)   |
{:.mbtablestyle}

## Repository Structure
* [Data Preparation (dataprep)](dataprep/): It contains Apache Spark data engineering pipelines to prepare the raw sources for model training.
* [Notebooks](notebooks/): It contains several notebooks with the plain experiments while developing the project, as well as a notebook showcasing the CLI usage.
* [Runtime environments (env)](env/): It contains conda and docker environment configuration recipies and files.

## System Requirements
* Java 1.8.x
* Apache Spark >= 2.3
* Conda >= 4.8.x
* Python 3.7.x
* CUDA 10.1
* Tensorflow 2.1.0
* Keras 2.3.1

## Workflow
### 1. Gather the data
* Data from twitter can be obtained from https://archive.org <a href="https://archive.org/details/twitterstream&tab=collection">here</a>.
* Data from Amazon product reviews can be obtained <a href="http://jmcauley.ucsd.edu/data/amazon/">here</a>.

### 2. Execute the data preparation pipelines
Both twitter and product review data needs to be pre-processed, for this there are two spark projects under [dataprep](dataprep/).
#### Tweets Transformer
This job takes the twitter data, filters out the tweets that are not geotagged and sinks the result as parquet files.
#### Amazon Product Reviews Transformer
This job takes the raw amazon product reviews and product metadata and converts them into 'documents' where each document has categories and either a review text or a product description.

### 3. Train the product classifier
Install the [cli](cli/) and follow the instructions to create the embeddings and train the classifier with the transformed product documents.

### 4. Predict the geographic area categories
Use the [cli](cli/) to predict the detect and classify the geographic
areas.

## Detailed process
A more detailed process is written in jupyter notebooks [here](notebooks/jupyter/)

## Docker
I have created a docker image with the environment and cli pre-installed and configured to run experments starting from pre-processed data https://hub.docker.com/r/ohtar10/wtsp.

To download the image:
```
docker image push ohtar10/wtsp:0.1.1
```

## License

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

See the LICENSE_ file in the root of this project for license details.
