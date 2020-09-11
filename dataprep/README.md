# Data preparation

In this folder you will find the data transformation pipelines used in this project before the actual training of the models. This is basically the Data Engineering part of the project.

## [Tweets Transformer](tweets-transformer)
This project contains a pipeline that takes the twitter data, filters out the tweets that are not geotagged and sinks the result as parquet.

## [Amazon Product Reviews Transformer](amz-product-review-transformer)
This project contains two pipelines:
* **Product Review Filter:** This pipeline will simply take the original datasets and filter the documents that comply with a specified set of product categories.
* **Product Review Transformer:** This pipeline will take a provided set of categories and product reviews and will transform then into documents ready for embedding training.

## [Scripts](scripts)
This directory contains simple python/pyspark scripts that are too simple for a full fledged project.