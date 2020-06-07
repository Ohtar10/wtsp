# Scripts

This folder contains simple python/pyspark scripts to perform specific tasks on datasets.

## [df-sampling](df-sampling.py)
This script is for taking stratified samples of bigger datasets.
### Usage:
```bash
$ spark-submit dataprep/scripts/df-sampling.py --help
20/06/07 10:38:55 WARN Utils: Your hostname, pop-os resolves to a loopback address: 127.0.1.1; using 192.168.1.110 instead (on interface wlp0s20f3)
20/06/07 10:38:55 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
20/06/07 10:38:55 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
usage: df-sampling.py [-h] [--fraction FRACTION] [--seed SEED] [--input INPUT]
                      [--output OUTPUT] [--class-col CLASS_COL]
                      [--split-char SPLIT_CHAR]

This program takes a stratified sample from the provided dataset.

optional arguments:
  -h, --help            show this help message and exit
  --fraction FRACTION   Fraction of sample
  --seed SEED           Random seed for reproducibility
  --input INPUT         Input path of the dataset to sample
  --output OUTPUT       Output path to store the result
  --class-col CLASS_COL
                        Category column to keep the strata proportions
  --split-char SPLIT_CHAR
                        Specifying this parameter indicates the class-col
                        contains multiple values separated by this character
log4j:WARN No appenders could be found for logger (org.apache.spark.util.ShutdownHookManager).
log4j:WARN Please initialize the log4j system properly.
log4j:WARN See http://logging.apache.org/log4j/1.2/faq.html#noconfig for more info.
```
Execution example:
```bash
$ spark-submit df-sample.py --fraction 0.2 --input <path to original dataset> --output <path to output folder> --class-col categories --split-char ";"
```
The above command will take a sample of 20% of the original dataset preserving the classes proportions.

## [transform-embeddings](transform-embeddings.py)
This script is for calculating and storing the document embeddings for
a dataset of documents using a pre-trained gensim Doc2Vec model. Along with the embeddings, it will also train a sklearn MultilabelBinarizer to encode the categories.
### Usage:
```bash
$ python ./transform-embeddings.py --help
usage: transform-embeddings.py [-h] --d2v-model D2V_MODEL --output OUTPUT
                               [--train-test-split] [--test-size TEST_SIZE]
                               documents_path

This program will transform the provided documents into embeddings using the
specified Doc2Vec model.

positional arguments:
  documents_path        Path to the raw documents

optional arguments:
  -h, --help            show this help message and exit
  --train-test-split    Perform a train test split before saving the results
  --test-size TEST_SIZE
                        Size of the test split if requrested, default to 0.3

required named arguments:
  --d2v-model D2V_MODEL
                        Path to the Gensim Doc2Vec model
  --output OUTPUT       Path to save the transformed embeddings
```
Execution example:
```bash
$ python transform-embeddings.py --d2v-model <path to a 'd2v_model.model'> --output <path to output folder> --train-test-split <path to original dataset>
```