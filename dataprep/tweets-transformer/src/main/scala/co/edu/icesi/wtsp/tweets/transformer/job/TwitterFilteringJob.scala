package co.edu.icesi.wtsp.tweets.transformer.job

import co.edu.icesi.wtsp.tweets.transformer.dataprep.TweetTransformerBuilder
import co.edu.icesi.wtsp.tweets.transformer.schema.Schemas
import co.edu.icesi.wtsp.tweets.transformer.spamfilter.TweetSpamAssassinPipeline
import co.edu.icesi.wtsp.tweets.transformer.statistics.TweetStatisticsCalculator
import org.apache.spark.sql.{DataFrame, SparkSession}

class TwitterFilteringJob(spark: Option[SparkSession], input: String,
                          output: String,
                          spamPipelineModel: String,
                          filterExpression: String)
  extends Job {

  override def execute(): Unit = {

    val sparkSession = spark.getOrElse(SparkSession.builder().getOrCreate())

    val tweets = TweetTransformerBuilder()
      .withCols(Schemas.tweetObject:_*)
      .build()
      .transform(sparkSession.read.json(input))

    //General statistics
    calculateStatistics(tweets, "full_stats")

    //Filter by those satisfying the given condition
    val tweetsWithCondition = tweets.where(filterExpression).cache()

    //Filtered statistics
    calculateStatistics(tweetsWithCondition, "condition_stats")

    //Predict if they are spam or ham
    val hamTweets = filterSpamTweets(tweetsWithCondition)

    //Ham statistics
    calculateStatistics(hamTweets, "final_stats")

    //Persist the final data frame to the provided output path
    hamTweets.write.mode("append")
      .partitionBy("year", "month", "day", "hour")
      .parquet(s"$output/tweets")
  }

  /**
    * Calculate tweet statistics for the given data frame
    * @param df tweets data frame
    * @param name name of the output for persistence.
    */
  private def calculateStatistics(df: DataFrame, name: String): Unit = {
    val statisticsCalculator = TweetStatisticsCalculator.calculateStatistics(df)
    statisticsCalculator.saveResults(s"$output/statistics/$name")
  }

  /**
    * Takes the given dataframe and calculates if they are spam or ham
    * and returns the tweets that were classified as ham
    * @param df original tweets
    * @return spam tweets data frame
    */
  private def filterSpamTweets(df: DataFrame): DataFrame = {
    TweetSpamAssassinPipeline(spamPipelineModel)
      .transform(df)
      .where("is_spam = 0")
  }

}

/**
  * Companion object for the
  * TwitterFilteringJob
  */
object TwitterFilteringJob{

  def apply(spark: Option[SparkSession] = None,
            input: String,
            output: String,
            spamPipelineModel: String,
            filterExpression: String = "place is not null and lang = 'en'"
            ): TwitterFilteringJob =
    new TwitterFilteringJob(spark, input, output, spamPipelineModel, filterExpression)

}
