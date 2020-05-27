from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, split, array_contains, concat_ws
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--fraction", help="Fraction of sample")
parser.add_argument("--seed", help="Random seed for reproducibility")
parser.add_argument("--input", help="Input path of the dataset to sample")
parser.add_argument("--output", help="Output path to sink the result")
parser.add_argument("--stratacol", help="Category column to keep the strata proportions")
parser.add_argument("--explodechar", help="Specifying this parameter indicates the stratacol contains multiple values separated by this character")
args = parser.parse_args()

if not args.fraction:
    sys.exit("The fraction is required")

if not args.input:
    sys.exit("The input path is required")

if not args.output:
    sys.exit("The output path is required")

if not args.stratacol:
    sys.exit("The strata column is required")


def simple_sampling(df):
    classes = list(map(lambda row: row[args.stratacol], df.select(args.stratacol).distinct().collect()))
    if seed:
        sample = df.filter(df[args.stratacol] == classes[0]).sample(fraction, seed)
    else:
        sample = df.filter(df[args.stratacol] == classes[0]).sample(fraction)

    for clazz in classes[1:]:
        if seed :
            sample = sample.union(df.filter(df[args.stratacol] == clazz).sample(fraction, seed))
        else:
            sample = sample.union(df.filter(df[args.stratacol] == clazz).sample(fraction))

    return sample, len(classes)


def explode_sampling(df):
    base_columns = df.columns
    columns = list(map(lambda c: split(col(c), args.explodechar).alias(c) if c == args.stratacol else col(c), base_columns))
    with_category_array = df.select(*columns)
    df = df.select(columns)
    classes = list(map(lambda row: row[args.stratacol], df.select(explode(args.stratacol).alias(args.stratacol)).distinct().collect()))

    if seed:
        sample = with_category_array.filter(array_contains(with_category_array[args.stratacol], classes[0])).sample(fraction, seed)
    else:
        sample = with_category_array.filter(array_contains(with_category_array[args.stratacol], classes[0])).sample(fraction)

    for clazz in classes[1:]:
        if seed :
            sample = sample.union(with_category_array.filter(array_contains(with_category_array[args.stratacol], clazz)).sample(fraction, seed))
        else:
            sample = sample.union(with_category_array.filter(array_contains(with_category_array[args.stratacol], clazz)).sample(fraction))

    select = list(map(lambda c: concat_ws(";", col(c)).alias(c) if c == args.stratacol else col(c), base_columns))
    return sample.select(select), len(classes)



spark = SparkSession.builder.appName("df-sampling").getOrCreate()
df = spark.read.parquet(args.input)
fraction = float(args.fraction)
seed = args.seed

if not args.explodechar:
    sample, partitions = simple_sampling(df)
else:
    sample, partitions = explode_sampling(df)

sample.coalesce(partitions).write.mode("overwrite").parquet(args.output)