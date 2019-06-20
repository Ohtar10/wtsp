# Tweets Transformer

This is a simple job that will filter tweets according to a given sql expression passed as argument to the job.

## Usage
```bash
spark-submit --class co.edu.icesi.wtsp.tweets.transformer.TwitterFilteringLocalApp \
tweets-transformer_2.11-0.0.1.jar <input> <output> <expression>
```