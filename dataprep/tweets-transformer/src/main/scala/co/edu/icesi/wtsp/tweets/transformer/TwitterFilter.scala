package co.edu.icesi.wtsp.tweets.transformer

import java.text.SimpleDateFormat

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

class TwitterFilter(spark: SparkSession, input: String, output: String){

  import spark.implicits._
  private val datePattern = "EEE MMM dd HH:mm:ss ZZZZZ yyyy"
  private val dateFormat = new SimpleDateFormat(datePattern)

  private def loadDataframe(): DataFrame = {
    val df = spark.read.json(input)
        .withColumn("created_timestamp", to_timestamp($"created_at", datePattern))
        .withColumn("created_year", year($"created_timestamp"))
        .withColumn("created_month", month($"created_timestamp"))
        .withColumn("created_day", dayofmonth($"created_timestamp"))
        .withColumn("created_hour", hour($"created_timestamp"))
    df
  }

  private def sinkDataframe(df: DataFrame): Unit = {
    df.write.mode("overwrite").parquet(output)
  }

  def filterWithExpression(expression: String): Unit = {
    val df = loadDataframe()
    val filtered = df.select("*").where(expression)
    sinkDataframe(filtered)
  }

  def string2Timestamp = udf((str: String) => {
    dateFormat.parse(str)
  })


}

object TwitterFilter {

  def apply(sparkSession: SparkSession, input: String, output: String): TwitterFilter = new TwitterFilter(sparkSession, input, output)

}
