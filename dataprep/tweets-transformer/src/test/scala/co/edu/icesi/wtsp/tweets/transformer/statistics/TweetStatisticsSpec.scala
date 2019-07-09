package co.edu.icesi.wtsp.tweets.transformer.statistics

import co.edu.icesi.wtsp.tweets.transformer.SpecCommon
import co.edu.icesi.wtsp.tweets.transformer.dataprep.TweetTransformerBuilder
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{FlatSpec, Matchers}

class TweetStatisticsSpec extends FlatSpec
    with Matchers
    with DataFrameSuiteBase
    with SpecCommon {

  def tweetsFixture = new {
    val tweets = spark.read.json(rawTweetsPath)
    val tweetsDF = TweetTransformerBuilder().build().transform(tweets)

    //For module calculated statistics
    val statistics = TweetStatisticsCalculator.calculateStatistics(tweetsDF)

    //For test calculated statistics
    val testStatistics = TestTweetStatistics.calculateStatistics(spark, tweets)
  }

  "The statistics module" should "count the total amount of tweets of the given data frame" in {
    val tf = tweetsFixture
    tf.statistics.tweetCount shouldBe tf.testStatistics.tweetCount
  }
  it should "count the total amount of distinct users" in {
    val tf = tweetsFixture
    tf.statistics.userCount shouldBe tf.testStatistics.userCount
  }
  it should "count the total amount of tweets matching a criteria"  in {
    fail("Not yet implemented")
  }
  it should "calculate the average tweets per month" in {
    fail("Not yet implemented")
  }
  it should "count the different locations from the total of tweets" in {
    fail("Not yet implemented")
  }
  it should "describe the different geographical locations" in {
    fail("Not yet implemented")
  }
  it should "count the tweets per geographical location" in {
    fail("Not yet implemented")
  }
  it should "describe the different location types" in {
    fail("Not yet implemented")
  }
  it should "count the tweets per location types" in {
    fail("Not yet implemented")
  }
  it should "count the tweets per language" in {
    fail("Not yet implemented")
  }

}

private object TestTweetStatistics {

  def calculateStatistics(spark: SparkSession, tweets: DataFrame): TweetStatistics = {
    val tweetCount = tweets.count()
    val userCount = tweets.select("user").distinct().count()
    TweetStatistics(tweetCount, userCount)
  }
}
