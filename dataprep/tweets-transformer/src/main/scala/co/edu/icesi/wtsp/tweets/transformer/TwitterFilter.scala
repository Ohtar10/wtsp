package co.edu.icesi.wtsp.tweets.transformer

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Twitter Filter
  * This class assumes the input directory contains files in json format
  * corresponding to tweet objects (https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object)
  * and applies a filter in sql clause form to them to finally sink them in the
  * provided output folder as parquet files.
  *
  * @param spark SparkSession object
  * @param input Path to the original files
  * @param output Path to the filtered results
  */
class TwitterFilter(spark: SparkSession, input: String, output: String){

  import spark.implicits._
  private val datePattern = "EEE MMM dd HH:mm:ss ZZZZZ yyyy"

  private def loadDataFrame(): DataFrame = {

    val df = spark.read.json(input)
        .withColumn("created_timestamp", to_timestamp($"created_at", datePattern))
        .withColumn("year", year($"created_timestamp"))
        .withColumn("month", month($"created_timestamp"))
        .withColumn("day", dayofmonth($"created_timestamp"))
        .withColumn("hour", hour($"created_timestamp"))
    df
  }

  private def sinkDataFrame(df: DataFrame): Unit = {
    df.write.mode("append")
      .partitionBy("year", "month", "day", "hour")
      .parquet(output)
  }

  /**
    * Applies the provided expression to the data frame created
    * from the input source and sink the results into the output
    * directory.
    *
    * @param fields the desired fields to get from the dataset
    * @param expression an sql expression to be applied for filtering
    */
  def filterWithExpression(fields: Array[String] = Array("*"), expression: String = ""): Unit = {
    val df = loadDataFrame()
    val filtered = if (expression.nonEmpty) df.select(fields.head, fields.tail:_*).where(expression) else df.select(fields.head, fields.tail:_*)
    sinkDataFrame(filtered)
  }

}

object TwitterFilter {

  def apply(sparkSession: SparkSession, input: String, output: String): TwitterFilter =
    new TwitterFilter(sparkSession, input, output)

}
