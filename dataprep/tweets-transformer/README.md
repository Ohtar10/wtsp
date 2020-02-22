# Tweets Transformer

This is a simple job that will filter tweets according to a given sql expression passed as argument to the job.

## Build
```bash
$ make all
```
This will compile the project from scratch, execute unit tests, package it and create the `dist` directory in the root folder in which you can find the distribution tar package. You can grab this package and expand it anywhere you want and execute the job using the bundled bash script.

## Usage
### Requirements
To execute this project the you need to have the following installed and configured in your local machine:
* Java 1.8.x
* Apache Spark version >= 2.3

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

### Execute a job
Depending on your configuration, defined variables and file locations
you can use the help to get further guidance. A general job execution
can be invoked like this:
```bash
./tweets-transformer.sh -o "/output/path" \
 -f "place is not null and lang = 'en'" \
 "/path/to/tweets_json/*/"
```

#### Warnings
* By default spark tries to store the events in `file:///tmp/spark-events/`, this directory must
exists before submitting the job.

### Executing against a cluster
To execute against a cluster, just use the Spark Related options according
to your needs.

#### Tips
* Be aware of your cluster resources to request them accordingly.
* Use the `time` unix command to measure how much it takes to execute the job.
* Spark logs are enabled by default use them to monitor your job; by default the logs are written to `/tmp/spark-event`
* Ensure your local Spark installation is properly configured to submit jobs against a YARN cluster.
* Remember that when submitting jobs to a cluster, the paths must correspond to HDFS or a distributed filesystem, not your local machine.
* After the job finishes, the Spark UI is shutdown, in order to see the stored events we must start
the spark history server: `$SPARK_HOME/sbin/start-history-server.sh`


```bash
time ./tweets-transformer.sh --master yarn \
 -o "/output/path" \
 -f "place is not null and lang = 'en'" \
 "/path/to/tweets_json/*/"
```