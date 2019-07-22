package co.edu.icesi.wtsp.tweets.transformer.job

import co.edu.icesi.wtsp.tweets.transformer.dataprep.TweetTransformerBuilder
import co.edu.icesi.wtsp.tweets.transformer.schema.Schemas
import co.edu.icesi.wtsp.tweets.transformer.spamfilter.TweetSpamAssassinPipeline
import co.edu.icesi.wtsp.tweets.transformer.statistics.TweetStatisticsCalculator
import co.edu.icesi.wtsp.util.JobLogging
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Twitter Filtering Job.
  *
  * This is the main class for the job, it controls
  * all the work needed for this job and invokes all
  * the corresponding parts to complete it.
  *
  * @param spark session
  * @param input path
  * @param output path
  * @param spamPipelineModel path to the spam pipeline model
  * @param filterExpression SQL filter expression
  */
class TwitterFilteringJob(spark: Option[SparkSession], input: String,
                          output: String,
                          spamPipelineModel: String,
                          filterExpression: String)
  extends Job with JobLogging{

  override def execute(): Unit = {

    val sparkSession = spark.getOrElse(SparkSession.builder().getOrCreate())

    logInfo(sparkSession, "Reading all the tweets")
    val tweets = TweetTransformerBuilder()
      .withCols(Schemas.tweetObject:_*)
      .build()
      .transform(sparkSession.read.schema(Schemas.sourceSchema).json(input))
      .cache()

    logInfo(sparkSession, "Calculating full statistics...")
    //General statistics
    calculateStatistics(tweets, "full_stats")

    logInfo(sparkSession, s"Filtering with expression: $filterExpression")
    //Filter by those satisfying the given condition
    val tweetsWithCondition = tweets.where(filterExpression).cache()

    //Filtered statistics
    logInfo(sparkSession, "Calculating statistics after filtering...")
    calculateStatistics(tweetsWithCondition, "condition_stats")

    //Predict if they are spam or ham
    logInfo(sparkSession, "Classifying tweets if they are spam or ham...")
    val hamTweets = filterSpamTweets(tweetsWithCondition)

    //Ham statistics
    logInfo(sparkSession, "Calculating statistics after getting rid of the spam tweets...")
    calculateStatistics(hamTweets, "final_stats")

    //Persist the final data frame to the provided output path
    logInfo(sparkSession, "Persisting final results in parquet files...")
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
    val statisticsCalculator = TweetStatisticsCalculator.calculateStatistics(df, name)
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
