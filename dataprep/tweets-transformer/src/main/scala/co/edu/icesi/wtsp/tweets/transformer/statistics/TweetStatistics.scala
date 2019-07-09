package co.edu.icesi.wtsp.tweets.transformer.statistics

import org.apache.spark.sql.DataFrame

case class TweetStatistics(tweetCount: Long,
                           userCount: Long)

/**
  * Contains a single function ''calculateStatistics''
  * which expects a tweet data frame conforming with the
  * schema given by Schemas.tweetObject and calculates
  * the basic statistics from it.
  *
  */
object TweetStatisticsCalculator {

  def calculateStatistics(tweets: DataFrame): TweetStatistics = {
    val tweetCount = tweets.count()
    TweetStatistics(tweetCount, 0)
  }

}