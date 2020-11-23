from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as types
import argparse
import sys
import re
import pickle
import os

parser = argparse.ArgumentParser(description="This program tokenizes documents in a provided data frame and saves the word index object for later usage.")
parser.add_argument("--input", help="Input path of the dataset", required=True)
parser.add_argument("--output", help="Output path to store the result", required=True)
parser.add_argument("--column", help="Column name with the document", required=True)
parser.add_argument("--vocab-size", type=int, help="Vocabulary max size", required=True)
parser.add_argument("--maxlen", type=int, help="Sequence max length", required=True)
parser.add_argument("--regex", default="[^\w+]", help="Regular expression to remove unwanted characters")
parser.add_argument("--partitions", default=1, type=int, help="Number of write partitions")
args = parser.parse_args()


def word_count(df, column, regex, vocab_size):
    return df.select(F.split(F.lower(F.col(column)), regex).alias('words'))\
    .select(F.explode(F.col('words')).alias('words'))\
    .filter("length(trim('words')) > 0")\
    .groupBy('words').count()\
    .orderBy('count', ascending=False)\
    .limit(vocab_size)

def build_word_index(word_count):
    word_index = word_count.withColumn('id', F.monotonically_increasing_id() + 1).select(F.col('words'), F.col('id'))
    word_index = word_index.rdd.map(lambda row: (row[0], row[1])).collect()
    return dict(word_index)

def tokenize(df, column, regex, word_index, maxlen):
    words_to_indices = F.udf(lambda words: [word_index[w] for w in words if w in word_index][:maxlen], types.ArrayType(types.IntegerType()))
    pad_sequence = F.udf(lambda words: [0] * (maxlen - len(words)) + words, types.ArrayType(types.IntegerType()))
    remove_from_regex = F.udf(lambda words: [re.sub(regex, '', w) for w in words], types.ArrayType(types.IntegerType()))
    return df.withColumn(f'tokenized_{column}', 
        pad_sequence(
            words_to_indices(
                remove_from_regex(
                    F.split(F.lower(F.col(column)), ' ')
                )
            )
        )
    )


spark = SparkSession.builder.appName("document-tokenizer").getOrCreate()
df = spark.read.parquet(args.input)
column = args.column
output = args.output
regex = args.regex
vocab_size = args.vocab_size
maxlen = args.maxlen
partitions = args.partitions

wc = word_count(df, column, regex, vocab_size)
wi = build_word_index(wc)
tokenized_docs = tokenize(df, column, regex, wi, maxlen)

# write the result
tokenized_docs.coalesce(partitions).write.parquet(output)

# write the word index
with open(os.path.join(output, 'word_index.pkl'), 'wb') as file:
    pickle.dump(wi, file)
