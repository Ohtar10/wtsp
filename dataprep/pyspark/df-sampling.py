from pyspark.sql import SparkSession
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--fraction", help="Fraction of sample")
parser.add_argument("--seed", help="Random seed for reproducibility")
parser.add_argument("--input", help="Input path of the dataset to sample")
parser.add_argument("--output", help="Output path to sink the result")
parser.add_argument("--stratacol", help="Category column to keep the strata proportions")
args = parser.parse_args()

if not args.fraction:
    sys.exit("The fraction is required")

if not args.input:
    sys.exit("The input path is required")

if not args.output:
    sys.exit("The output path is required")

if not args.stratacol:
    sys.exit("The strata column is required")

spark = SparkSession.builder.appName("df-sampling").getOrCreate()

df = spark.read.parquet(args.input)
classes = list(map(lambda row: row[args.stratacol], df.select(args.stratacol).distinct().collect()))

fraction = float(args.fraction)
seed = args.seed
if seed :
    sample = df.filter(df[args.stratacol] == classes[0]).sample(fraction, seed)
else:
    sample = df.filter(df[args.stratacol] == classes[0]).sample(fraction)

for clazz in classes:
    if seed :
        sample = df.filter(df[args.stratacol] == clazz).sample(fraction, seed)
    else:
        sample = df.filter(df[args.stratacol] == clazz).sample(fraction)

sample.coalesce(len(classes)).write.mode("overwrite").parquet(args.output)