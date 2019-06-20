package co.edu.icesi.wtsp.tweets.transformer

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

class TwitterFilter(spark: SparkSession, input: String, output: String){

  import spark.implicits._
  private val datePattern = "EEE MMM dd HH:mm:ss ZZZZZ yyyy"

  private def loadDataframe(): DataFrame = {
    //to read from hadooop recursively
    //spark.sparkContext.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive", "true")

    val df = spark.read.json(input)
        .withColumn("created_timestamp", to_timestamp($"created_at", datePattern))
        .withColumn("year", year($"created_timestamp"))
        .withColumn("month", month($"created_timestamp"))
        .withColumn("day", dayofmonth($"created_timestamp"))
        .withColumn("hour", hour($"created_timestamp"))
    df
  }

  private def sinkDataframe(df: DataFrame): Unit = {
    df.write.mode("append")
      .partitionBy("year", "month", "day", "hour")
      .parquet(output)
  }

  def filterWithExpression(expression: String): Unit = {
    val df = loadDataframe()
    val filtered = df.select("*").where(expression)
    sinkDataframe(filtered)
  }

}

object TwitterFilter {

  def apply(sparkSession: SparkSession, input: String, output: String): TwitterFilter =
    new TwitterFilter(sparkSession, input, output)

}
