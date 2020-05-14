# Amazon Product Reviews Transformer

This job transforms the Amazon product reviews and metadata [dataset](http://jmcauley.ucsd.edu/data/amazon/), to create category-corpus documents.

## Requirements
To execute this project the you need to have the following installed and configured in your local machine:
* Java 1.8
* Scala 2.11.x
* Sbt 0.13.16
* Apache Spark >= 2.3

## Build
```bash
$ make all
```
This will compile the project from scratch, execute unit tests, package it and create the `dist` directory in the root folder in which you can find the distribution tar package. You can grab this package and expand it anywhere you want and execute the job using the bundled bash script.

## Usage
After expanding the build package, enter into the folder and use the `product-doc-transformer.sh` bash script as follows:

```bash
$ cd <path where dist package was expanded>
$ ./product-doc-transformer.sh --help

                     _            _                         
 _ __  _ __ ___   __| |_   _  ___| |_                       
| '_ \| '__/ _ \ / _` | | | |/ __| __|                      
| |_) | | | (_) | (_| | |_| | (__| |_                       
| .__/|_|  \___/ \__,_|\__,_|\___|\__|                      
|_|                                                         
     _                                       _              
  __| | ___   ___ _   _ _ __ ___   ___ _ __ | |_ ___        
 / _` |/ _ \ / __| | | | '_ ` _ \ / _ \ '_ \| __/ __|       
| (_| | (_) | (__| |_| | | | | | |  __/ | | | |_\__ \       
 \__,_|\___/ \___|\__,_|_| |_| |_|\___|_| |_|\__|___/       
                                                            
 _                        __                                
| |_ _ __ __ _ _ __  ___ / _| ___  _ __ _ __ ___   ___ _ __ 
| __| '__/ _` | '_ \/ __| |_ / _ \| '__| '_ ` _ \ / _ \ '__|
| |_| | | (_| | | | \__ \  _| (_) | |  | | | | | |  __/ |   
 \__|_|  \__,_|_| |_|___/_|  \___/|_|  |_| |_| |_|\___|_| 

Wrapper script to execute the Apache Spark Job that transforms
tweets from their original json into parquet format and some
constraints.

Usage:
  product-doc-transformer.sh [--options]
  product-doc-transformer.sh -h | --help

Options:
  -h --help               Display this help information.

Spark Related Options:
  -sh --spark-home        Spark home path (Default: environment variable SPARK_HOME)
  -m  --master            Spark master (Default: 'local[*]')
  -dm --driver-memory     Spark driver memory (Default: 16G)
  -em --executor-memory   Spark executor memory (Default: 7G)
  -dc --driver-cores      Spark driver cores (Default: 12)
  -ec --executor-cores    Spark executor cores (Default: 45)
  -ne --num-executors     Spark number of executors (Default: 12)
  -el --event-log         Location to save the spark event logs (Default: /tmp/spark-event)
  -jj --job-jar           Spark Job Jar path (Default: ./product-doc-transformer.jar)

Job Related Options:
  -o  --outputs           Output directories in k1=v1,k2=v2 format (keys: full-documents-output, review-documents-output, metadata-documents-output, filter-output [Required at least one.]
  -md --metadata          Product metadata file path [Required.]
  -rv --reviews           Product reviews file path [Required.]
  -a --metadata-cols      The columns to select from the metadata.
  -b --review-cols        The columns to select from the reviews.
  -cm --category-map      Product category mapping file (Default: ./category_mappings.yml)
  -st --steps             The pipeline steps to execute (Default: filter, transform-metadata, transform-reviews, aggregate-documents)
  -l  --limit             Maximum number of records to process
  -s  --seed              Random seed for sampling



```

### Execute the pipeline
Depending on your configuration, defined variables and file locations the command below may vary. You can use the help to get further guidance. A general job execution can be invoked like this:

```bash
./product-doc-transformer.sh \ 
-o full-documents-output=/path/to/documents/output \ 
-md /path/to/product/metadata.json \ 
-rv /path/to/product/reviews.json
```

#### Warnings
* By default spark tries to store the events in `file:///tmp/spark-event/`, this directory must exist before submitting the job.

### Executing against a cluster
To execute against a cluster, just use the Spark Related options according to your needs.

#### Tips
* Be aware of your cluster resources to request them accordingly.
* Spark logs are enabled by default use them to monitor your job; by default the logs go to `/tmp/spark-event`
* Ensure your local Spark installation is properly configured to submit jobs against a YARN cluster.
* Remember that when submitting jobs to a cluster, the paths must correspond to HDFS or a distributed filesystem, not your local machine.
* After the job finishes, the Spark UI is shutdown, in order to see the stored events we must start
the spark history server: `$SPARK_HOME/sbin/start-history-server.sh <path-to-spark-events>`
