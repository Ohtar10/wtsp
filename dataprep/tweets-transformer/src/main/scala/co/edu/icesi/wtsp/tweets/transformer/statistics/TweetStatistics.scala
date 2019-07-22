package co.edu.icesi.wtsp.tweets.transformer.statistics

import co.edu.icesi.wtsp.util.JobLogging
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

case class TweetStatistics(tweetCount: Long,
                           userCount: Long,
                           avgTweetsPerMonth: Double,
                           locationCount: Long,
                           tweetCountByLocationType: Map[String, Long],
                           tweetCountByLanguage: Map[String, Long],
                           tweetCountByCountry: Map[String, Long]) {

  def saveResults(path: String): Unit = {
    val spark = SparkSession.builder().getOrCreate()

    val df = spark.createDataFrame(List(this))

    df.write.mode("overwrite").json(path)
  }

}

/**
  * Contains a single function ''calculateStatistics''
  * which expects a tweet data frame conforming with the
  * schema given by Schemas.tweetObject and calculates
  * the basic statistics from it.
  *
  */
object TweetStatisticsCalculator extends JobLogging{

  def calculateStatistics(tweets: DataFrame, name: String = ""): TweetStatistics = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    logInfo(spark, "Counting tweets...", Option(name))
    val tweetCount = tweets.count()
    logInfo(spark, "Counting the different users...", Option(name))
    val userCount = tweets.select("user_id").distinct().count()

    logInfo(spark, "Calculating the average tweets per month", Option(name))
    val avgTweetsPerMonth = tweets.withColumn("created_month",
          date_format($"created_timestamp", "yyyy-MM"))
      .groupBy($"created_month")
      .agg(count(lit(1)).as("tweets_per_month"))
      .agg(avg($"tweets_per_month")).first().getDouble(0)

    logInfo(spark, "Counting the different locations...", Option(name))
    val locationCount = tweets.select("place_name").distinct().count()

    logInfo(spark, "Counting the tweets grouped by place type...", Option(name))
    val tweetCountByLocationType = tweets.groupBy("place_type")
      .agg(count(lit(1)).as("count"))
      .select(
        coalesce($"place_type", lit("undefined")),
        $"count")
      .collect().map( r => (r.getString(0), r.getLong(1))).toMap

    logInfo(spark, "Counting the tweets by language...", Option(name))
    val tweetCountByLanguage = tweets.groupBy("lang")
      .agg(count(lit(1)).as("count"))
      .select(
        coalesce($"lang", lit("undefined")),
        $"count")
      .collect().map( r => (r.getString(0), r.getLong(1))).toMap

    logInfo(spark, "Counting the tweets by country...", Option(name))
    val tweetCountByCountry = tweets.groupBy("country")
        .agg(count(lit(1)).as("count"))
        .select(
          coalesce($"country", lit("undefined")),
          $"count")
        .collect().map(r => r.getString(0) -> r.getLong(1)).toMap

    TweetStatistics(tweetCount,
      userCount,
      avgTweetsPerMonth,
      locationCount,
      tweetCountByLocationType,
      tweetCountByLanguage,
      tweetCountByCountry)
  }

}