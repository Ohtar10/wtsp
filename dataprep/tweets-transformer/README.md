# Tweets Transformer

This is a simple job that will filter tweets according to a given sql expression passed as argument to the job.

## Usage
```bash
spark-submit --class co.edu.icesi.wtsp.tweets.transformer.TwitterFilteringLocalApp \
tweets-transformer_2.11-0.2.0.jar -i <input> -o <output> -s <spam-pipeline-path> -f <filter-expression>
```

## Example
```bash
spark-submit --class co.edu.icesi.wtsp.tweets.transformer.TwitterFilteringLocalApp \
 /path/to/tweets-transformer-with-deps-0.2.0.jar \
 -i "/path/to/tweets_json/*/" -o "/output/path" \
 -s "/path/to/spam/pipeline/model" \
 -f "place is not null and lang = 'en'"
```

## Executing for real work
For real work, we should provide additional properties and options so we
can track properly the progress, time and persist the events for later exploration
when the job finishes. We should then execute the command like this:

```bash
time spark-submit --driver-memory <xG> --executor-memory <xG> --driver-cores <NUM> \
--executor-cores <NUM> --num-executors <NUM> --conf "spark.eventLog.enabled=true" \ 
--conf  "spark.eventLog.compress=true" tweets-transformer_2.11-0.2.0.jar -i <input> \ 
-o <output> -s <spam-pipeline-path> -f <filter-expression>
```

### Explanation
* `time` help us to see how much the job took to complete at OS level.
* Depending on the environment, we might want to fine control the driver and executor
memory, cores and quantity.
* `spark.eventLog.enabled=true` makes the spark events from the job to be persisted beyond
the life of the job. After the job finishes, the spark-ui is shutdown. This property persists the
events to later be viewed in the spark history server.
* `spark.eventLog.compress=true` makes the spark events resulting from an executed job to be compressed

After the job finishes, the Spark UI is shutdown, in order to see the stored events we must start
the spark history server: `$SPARK_HOME/sbin/start-history-server.sh`

#### Warnings
* By default spark tries to store the events in `file:///tmp/spark-events/`, this directory must
exists before submitting the job.