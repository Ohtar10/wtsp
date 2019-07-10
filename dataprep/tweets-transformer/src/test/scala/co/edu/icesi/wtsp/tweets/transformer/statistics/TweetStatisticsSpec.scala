package co.edu.icesi.wtsp.tweets.transformer.statistics

import java.io.File
import java.nio.file.{Files, Paths}

import co.edu.icesi.wtsp.tweets.transformer.SpecCommon
import co.edu.icesi.wtsp.tweets.transformer.dataprep.TweetTransformerBuilder
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import io.circe.syntax._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{FlatSpec, Matchers}

class TweetStatisticsSpec extends FlatSpec
    with Matchers
    with DataFrameSuiteBase
    with SpecCommon {

  private def tweetsFixture = new {
    import spark.implicits._
    val datePattern = "EEE MMM dd HH:mm:ss ZZZZZ yyyy"

    val tweets = spark.read.json(rawTweetsPath)
      .withColumn("created_timestamp", to_timestamp($"created_at", datePattern))
      .withColumn("year", year($"created_timestamp"))
      .withColumn("month", month($"created_timestamp"))
      .withColumn("day", dayofmonth($"created_timestamp"))
      .withColumn("hour", hour($"created_timestamp"))
      .cache()

    val tweetsDF = TweetTransformerBuilder().build().transform(tweets)

    //For module calculated statistics
    val statistics = TweetStatisticsCalculator.calculateStatistics(tweetsDF)

    //For test calculated statistics
    val testStatistics = TestTweetStatistics.calculateStatistics(spark, tweets)
  }

  "The statistics module" should "calculate the statistics correctly" in {
    val tf = tweetsFixture
    tf.statistics.tweetCount shouldBe tf.testStatistics.tweetCount
    tf.statistics.userCount shouldBe tf.testStatistics.userCount
    tf.statistics.avgTweetsPerMonth shouldBe tf.testStatistics.avgTweetsPerMonth
    tf.statistics.locationCount shouldBe tf.testStatistics.locationCount
    tf.statistics.tweetCountByLocationType shouldBe tf.testStatistics.tweetCountByLocationType
    tf.statistics.tweetCountByLanguage shouldBe tf.testStatistics.tweetCountByLanguage
    tf.statistics.tweetCountByCountry shouldBe tf.testStatistics.tweetCountByCountry
  }
  it can "be converted to persist as a json file" in {
    val tf = tweetsFixture
    val output = testOutputPath + "statistics"
    tf.statistics.saveResults(output)

    Files.exists(Paths.get(output)) shouldBe true

    val statsDF = spark.read.json(output)
    statsDF shouldBe a[DataFrame]
    statsDF.columns.toSet shouldBe Set("tweetCount",
    "userCount",
    "avgTweetsPerMonth",
    "locationCount",
    "tweetCountByLocationType",
    "tweetCountByLanguage",
    "tweetCountByCountry")

    deleteRecursively(new File(testOutputPath))
  }

}

private object TestTweetStatistics {

  def calculateStatistics(spark: SparkSession, tweets: DataFrame): TweetStatistics = {
    import spark.implicits._

    tweets.createOrReplaceTempView("tweets")

    val tweetCount = tweets.count()
    val userCount = tweets.select("user.id").distinct().count()
    val avgTweetsPerMonth = spark.sql("select avg(tweets_per_month) as avg_tweets_per_month from (" +
      "select date_format(created_timestamp, \"yyyy-MM\") as created_month, count(1) as tweets_per_month " +
      "from tweets group by created_month)").first().getDouble(0)

    val locationCount = tweets.select("place.name").distinct().count()

    val tweetCountByLocationType = tweets.groupBy("place.place_type")
        .agg(count(lit(1)).as("count"))
        .select(
          coalesce($"place_type", lit("undefined")),
          $"count")
        .collect().map( r => r.getString(0) -> r.getLong(1)).toMap

    val tweetCountByLanguage = tweets.groupBy("lang")
        .agg(count(lit(1)).as("count"))
        .select(
          coalesce($"lang", lit("undefined")),
          $"count")
        .collect().map(r => r.getString(0) -> r.getLong(1)).toMap

    val tweetCountByCountry = tweets.groupBy("place.country")
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
      tweetCountByCountry
    )
  }
}
