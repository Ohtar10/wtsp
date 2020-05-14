# Tweets Transformer

This is a simple job that filters tweets according to a given sql expression passed as argument to the job.

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
### Requirements
After expanding the build package, enter the folder and use the `tweets-transformer.sh` bash script as follows:

```bash
$ cd <path where dist package was expanded>
$ ./tweets-transformer.sh --help

 _                     _         _                        __                                
| |___      _____  ___| |_ ___  | |_ _ __ __ _ _ __  ___ / _| ___  _ __ _ __ ___   ___ _ __ 
| __\ \ /\ / / _ \/ _ \ __/ __| | __| '__/ _` | '_ \/ __| |_ / _ \| '__| '_ ` _ \ / _ \ '__|
| |_ \ V  V /  __/  __/ |_\__ \ | |_| | | (_| | | | \__ \  _| (_) | |  | | | | | |  __/ |   
 \__| \_/\_/ \___|\___|\__|___/  \__|_|  \__,_|_| |_|___/_|  \___/|_|  |_| |_| |_|\___|_| 

Wrapper script to execute the Apache Spark Job that transforms
tweets from their original json into parquet format and some
constraints.

Usage:
  tweets-transformer.sh [--options] INPUT_PATH
  tweets-transformer.sh -h | --help

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
  -jj --job-jar           Spark Job Jar path (Default: ./tweets-transformer.jar)

Job Related Options:
  -o  --output            Output directory [Required.]
  -f --filter             Filter expression for the tweets (Default: 'place is not null')

```

### Execute the pipeline
Depending on your configuration, defined variables and file locations the command below may vary. You can use the help to get further guidance. A general job execution
can be invoked like this:

```bash
./tweets-transformer.sh \ 
-o "/output/path" \ 
-f "place is not null and lang = 'en'" \ 
"/path/to/json_tweets/*/"
```

#### Warnings
* By default spark tries to store the events in `file:///tmp/spark-event/`, this directory must exist before submitting the job.

### Executing against a cluster
To execute against a cluster, just use the Spark Related options according
to your needs.

### Tips
* When working with the datasets downloaded from archive.org, you'll need to specify wildcards in the input data paths. For example, the input data has the following folder structure:
```bash
2013 (year)
|-09 (month)
  |-01 (date)
    |-00 (hour)
    |-02 (hour)
    ...
    |-23 (hour)
  |-02 (date)
    |-00 
    ...
    |-23
  ...
  |-31
```
You will need to specify the input path as `<base_path>/2013/*/*/*/*`. This means that spark needs to traverse all the directories under 2013, and the subsequent directories under the month directory, and so on.
* Beware of your cluster resources to request them accordingly.
* Spark logs are enabled by default use them to monitor your job; by default the logs go to `/tmp/spark-event`
* Ensure your local Spark installation is properly configured to submit jobs against a YARN cluster.
* Remember that when submitting jobs to a cluster, the paths must correspond to HDFS or a distributed filesystem, not your local machine.
* After the job finishes, the Spark UI is shutdown, in order to see the stored events we must start
the spark history server: `$SPARK_HOME/sbin/start-history-server.sh <path-to-spark-events>`