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
parser.add_argument("--split-regex", default="\s+", help="Regular expression to split words in a sequence")
parser.add_argument("--clean-regex", default="[?!\(\)\"'#@\$%\^&\*\-\_|+=\[\{\}\],\.\?/<>~¿]", help="Regular expression to remove unwanted characters")
parser.add_argument("--batches", default=0, type=int, help="Number of batches to write")
parser.add_argument("--batch-size", default=0, type=int, help="Number of records per batch")
args = parser.parse_args()

if args.batches > 0 and args.batch_size > 0:
    sys.exit(f'Error: Please, specify either number of batches or batch size but not both')

spark = SparkSession.builder.appName("document-tokenizer").getOrCreate()
df = spark.read.parquet(args.input)
column = args.column
output = args.output
split_regex = args.split_regex
clean_regex = args.clean_regex
vocab_size = args.vocab_size
maxlen = args.maxlen

def split_document_fn(document):
    words = re.split(split_regex, document.lower())
    words = [re.sub(clean_regex, '', word) for word in words]
    return [word.strip() for word in words if len(word.strip()) > 0]

split_document = F.udf(split_document_fn, types.ArrayType(types.StringType()))

def word_count(df, column, split_regex, clean_regex, vocab_size):
    return df.select(split_document(F.col(column)).alias('words'))\
    .select(F.explode(F.col('words')).alias('words'))\
    .groupBy('words').count()\
    .orderBy('count', ascending=False)\
    .limit(vocab_size)

def build_word_index(word_count):
    word_index = word_count.withColumn('id', F.monotonically_increasing_id()).select(F.col('words'), F.col('id'))
    word_index = word_index.rdd.map(lambda row: (row[0], row[1])).collect()
    return dict(word_index)

def tokenize(df, column, word_index, maxlen):
    words_to_indices = F.udf(lambda words: [word_index[w] for w in words if w in word_index][:maxlen], types.ArrayType(types.IntegerType()))
    pad_sequence = F.udf(lambda words: [0] * (maxlen - len(words)) + words, types.ArrayType(types.IntegerType()))
    return df.withColumn(f'tokenized_{column}', 
        pad_sequence(
            words_to_indices(
                split_document(F.col(column))
            )
        )
    )

wc = word_count(df, column, split_regex, clean_regex, vocab_size)
wi = build_word_index(wc)
tokenized_docs = tokenize(df, column, wi, maxlen)

# write the result
if args.batches > 0:
    tokenized_docs.coalesce(args.batches).write.parquet(output)
elif args.batch_size > 0:
    total = tokenized_docs.count()
    batches = total // args.batch_size
    print(f"Number of batches to write: {batches}")
    tokenized_docs.repartition(batches).write.parquet(output)
    print("Finished writing parquet files")
else:
    sys.exit(f'Error: please provide either the number of output batches or a batch size')

# write the word index
with open(os.path.join(output, 'word_index.pkl'), 'wb') as file:
    pickle.dump(wi, file)

print("Finished writing the word index")
print("Tokenization finished")
