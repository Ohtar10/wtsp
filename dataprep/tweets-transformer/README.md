# Tweets Transformer

This is a simple job that will filter tweets according to a given sql expression passed as argument to the job.

## Usage
```bash
spark-submit --class co.edu.icesi.wtsp.tweets.transformer.TwitterFilteringLocalApp \
tweets-transformer_2.11-0.2.0.jar -i <input> -o <output> -s <spam-pipeline-path> -f <filter-expression>
```

## ExaMPLE
```bash
spark-submit --class co.edu.icesi.wtsp.tweets.transformer.TwitterFilteringLocalApp \
 /path/to/tweets-transformer-with-deps-0.2.0.jar \
 -i "/path/to/tweets_json/*/" -o "/output/path" \
 -s "/path/to/spam/pipeline/model" \
 -f "place is not null and lang = 'en'"
```