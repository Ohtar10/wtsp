#!/usr/bin/bash

### Path variables
spam_pipeline="/user/LFERRO/models/spam/tweet-spam-assassin-tf"
input="/user/LFERRO/datasets/twitter/2012/04/*/*"
output="/user/LFERRO/outputs/twitter/"
job_jar_path="/home/LFERRO/git/wtsp/dataprep/tweets-transformer/target/scala-2.11/tweets-transformer-with-deps-0.2.0.jar"
spark_eventlog_path="/tmp/spark-event-logs/"

### Spark infrastructure variables
driver_memory="4G"
executor_memory="7G"
driver_cores="8"
executor_cores="45"
num_executors="12"

### Job variables
filter_expresion="place is not null and lang = 'en'"

time $SPARK_HOME/bin/spark-submit --class co.edu.icesi.wtsp.tweets.transformer.TwitterFilteringApp \
--master yarn \
--driver-memory ${driver_memory} \
--executor-memory ${executor_memory} \
--driver-cores ${driver_cores} \
--total-executor-cores ${executor_cores} \
--num-executors ${num_executors} \
--conf "spark.eventLog.enabled=true" \
--conf "spark.eventLog.compress=true" \
--conf "spark.eventLog.dir=${spark_eventlog_path}" \
${job_jar_path} -i ${input} -o ${output} -s ${spam_pipeline} -f "${filter_expresion}"