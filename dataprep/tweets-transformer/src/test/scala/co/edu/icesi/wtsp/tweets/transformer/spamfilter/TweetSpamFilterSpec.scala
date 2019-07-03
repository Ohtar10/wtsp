package co.edu.icesi.wtsp.tweets.transformer.spamfilter

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._
import org.mockito.MockitoSugar
import org.scalatest.{FlatSpec, Matchers}

/**
  * Tweet Spam Filter Spec
  *
  * This class contains all the testing needed to
  * validate the tweet spam filter functionality
  * based on a pre-trained spam filter model.
  *
  */
class TweetSpamFilterSpec extends FlatSpec
  with MockitoSugar
  with Matchers
  with DataFrameSuiteBase{

  private[spamfilter] val modelPath = "src/test/resources/models/spark/tweet-spam-assassin"
  private[spamfilter] val tweetsPath = "src/test/resources/models/tweets/test.csv"
  private[spamfilter] val rawTweetsPath = "src/test/resources/tweets/*/"

  "The spam filter pipeline" should "be able to transform the original tweet object into model's input" in {
    val spamInputPipeline = TweetSpamInputPipeline()
    val rawTweetsDF = spark.read.json(rawTweetsPath)

    val schema = new StructType()
      .add("Id", LongType, nullable = false)
      .add("Tweet",StringType, nullable = false)
      .add("following",DoubleType, nullable = true)
      .add("followers",DoubleType, nullable = true)
      .add("actions", DoubleType, nullable = true)
      .add("is_retweet", DoubleType, nullable = true)
      .add("location", StringType, nullable = true)
      .add("Type", StringType, nullable = false)

    val transformedTweets = spamInputPipeline.transform(rawTweetsDF)

    transformedTweets.schema shouldBe schema
  }

  "The spam filter model" can "be loaded as a spark model from file system" in {
    val model = CrossValidatorModel.load(modelPath)
    model shouldBe a[CrossValidatorModel]
  }

  it should "be able to be used to predict a new entries" in {
    val model: CrossValidatorModel = CrossValidatorModel.load(modelPath)
    val spamInputPipeline = TweetSpamInputPipeline()

    val tweetsDF = spamInputPipeline.transform(spark.read.csv(tweetsPath))

    val predictions = model.transform(tweetsDF)

    predictions shouldBe a[DataFrame]
    predictions.columns should contain "prediction"
    predictions.select("prediction").distinct().collect() shouldEqual Array(0.0, 1.0)
  }

}
