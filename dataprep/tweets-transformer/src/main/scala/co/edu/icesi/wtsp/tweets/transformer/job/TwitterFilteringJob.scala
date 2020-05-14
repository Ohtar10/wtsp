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
class TwitterFilteringJob(spark: SparkSession, input: String,
                          output: String,
                          spamPipelineModel: Option[String],
                          filterExpression: Option[String])
  extends Job with JobLogging{

  override def execute(): Unit = {

    logInfo(spark, "Reading all the tweets")
    val tweets = TweetTransformerBuilder()
      .withCols(Schemas.tweetObject:_*)
      .build()
      .transform(spark.read.schema(Schemas.sourceSchema).json(input))
      .cache()

    val tweetsWithCondition = filterExpression match {
      case Some(expression) => {
        logInfo(spark, s"Filtering with expression: $expression")

        tweets.where(expression)
      }
      case None => tweets
    }

    tweetsWithCondition.cache()

    val finalResults: DataFrame = spamPipelineModel match {
      case Some(pipelinePath) => {

        //Predict if they are spam or ham
        logInfo(spark, "Classifying tweets if they are spam or ham...")
        val hamTweets = filterSpamTweets(tweetsWithCondition, pipelinePath)

        hamTweets
      }
      case None => {
        tweetsWithCondition
      }
    }

    // Decide if further statistics calculation is required

    // Full statistics if any condition is informed
    if (spamPipelineModel.isDefined || filterExpression.isDefined){
      calculateStatistics(tweets,
        "full_stats",
        Option("Calculating full statistics..."))
    }

    // Condition stats only if both are informed
    if (spamPipelineModel.isDefined && filterExpression.isDefined){
      calculateStatistics(tweetsWithCondition,
        "condition_stats",
        Option("Calculating statistics after filtering..."))
    }

    //Final statistics
    calculateStatistics(finalResults,
      "final_stats",
      Option("Calculating statistics final statistics"))

    //Persist the final data frame to the provided output path
    logInfo(spark, "Persisting final results in parquet files...")
    finalResults.coalesce(1).write.mode("append")
      .partitionBy("year", "month", "day", "hour")
      .parquet(s"$output/tweets")
  }

  /**
    * Calculate tweet statistics for the given data frame
    * @param df tweets data frame
    * @param name name of the output for persistence.
    */
  private def calculateStatistics(df: DataFrame, name: String, logMessage: Option[String] = None): Unit = {
    logMessage.foreach(logInfo(spark, _))
    val statisticsCalculator = TweetStatisticsCalculator.calculateStatistics(df, name)
    statisticsCalculator.saveResults(s"$output/statistics/$name")
  }

  /**
    * Takes the given dataframe and calculates if they are spam or ham
    * and returns the tweets that were classified as ham
    * @param df original tweets
    * @return spam tweets data frame
    */
  private def filterSpamTweets(df: DataFrame, pipelinePath: String): DataFrame = {
    TweetSpamAssassinPipeline(pipelinePath)
      .transform(df)
      .where("is_spam = 0")
  }

}

/**
  * Companion object for the
  * TwitterFilteringJob
  */
object TwitterFilteringJob{

  def apply(spark: SparkSession,
            input: String,
            output: String,
            spamPipelineModel: Option[String] = None,
            filterExpression: Option[String] = None
            ): TwitterFilteringJob =
    new TwitterFilteringJob(spark, input, output, spamPipelineModel, filterExpression)

}
