package co.edu.icesi.wtsp.tweets.transformer.spamfilter

import co.edu.icesi.wtsp.tweets.transformer.SpecCommon
import co.edu.icesi.wtsp.tweets.transformer.dataprep.TweetTransformerBuilder
import co.edu.icesi.wtsp.tweets.transformer.schema.Schemas
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{HashingTF, Word2VecModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
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
class TweetSpamAssassinSpec extends FlatSpec
  with MockitoSugar
  with Matchers
  with DataFrameSuiteBase
  with SpecCommon{

  "The spark spam filter pipeline" should "be able to load as a tf based pipeline model" in {
    val pipeline = PipelineModel.load(tfPipeline)

    pipeline shouldBe a[PipelineModel]
    pipeline.stages.length shouldBe 6
    pipeline.stages(3) shouldBe a[HashingTF]
  }
  it should "be able to load as a word2vec based pipeline model" in {
    val pipeline = PipelineModel.load(w2vPipeline)

    pipeline shouldBe a[PipelineModel]
    pipeline.stages.length shouldBe 5
    pipeline.stages(2) shouldBe a[Word2VecModel]
  }

  "The tf pipeline" should "be able to transform new entries" in {
    val pipeline = PipelineModel.load(tfPipeline)
    val tweetsDF = TweetsUtil.prepareTweets(spark, tweetsPath)

    val predictions = pipeline.transform(tweetsDF)

    predictions.count() shouldBe tweetsDF.count()
    predictions.columns should contain ("prediction")
    val predValues = predictions.select("prediction").distinct().collect().map(r => r.getAs[Double](0))
    predValues.length shouldBe 2
    predValues should contain (0.0)
    predValues should contain (1.0)
  }

  "The word2vec pipeline" should "be able to transform new entries" in {
    val pipeline = PipelineModel.load(w2vPipeline)
    val tweetsDF = TweetsUtil.prepareTweets(spark, tweetsPath)

    val predictions = pipeline.transform(tweetsDF)

    predictions.count() shouldBe tweetsDF.count()
    predictions.columns should contain ("prediction")
    val predValues = predictions.select("prediction").distinct().collect().map(r => r.getAs[Double](0))
    predValues.length shouldBe 2
    predValues should contain (0.0)
    predValues should contain (1.0)
  }

  "The spam filter transformer" should "be able to transform from raw tweets with tf pipeline" in {
    val tweetSpamAssassin = TweetSpamAssassinPipeline(tfPipeline)
    val rawTweets = spark.read.json(rawTweetsPath)

    val dataPrepTransformer = TweetTransformerBuilder()
      .withCols(Schemas.tweetObject:_*)
      .build()

    val preparedTweets = dataPrepTransformer.transform(rawTweets)

    val transformed = tweetSpamAssassin.transform(preparedTweets)

    rawTweets.count() should be > transformed.count()
    transformed.count() should be > 0L
    transformed.columns should contain("is_spam")
  }

  it should "be able to transform from raw tweets with word2vec pipeline" in {
    val tweetSpamAssassin = TweetSpamAssassinPipeline(w2vPipeline)
    val rawTweets = spark.read.json(rawTweetsPath)

    val dataPrepTransformer = TweetTransformerBuilder()
      .withCols(Schemas.tweetObject:_*)
      .build()

    val preparedTweets = dataPrepTransformer.transform(rawTweets)

    val transformed = tweetSpamAssassin.transform(preparedTweets)

    rawTweets.count() should be > transformed.count()
    transformed.count() should be > 0L
    transformed.columns should contain("is_spam")
  }

}

/**
  * This is an utility class that prepares the
  * tweet file for predicting if the instances
  * are spam or ham.
  *
  * This is only valid within this Spec since
  * the internal pipeline will need to take
  * responsibility of transforming the raw tweets
  * into the format the pipeline expects.
  *
  */
private object TweetsUtil{

  def prepareTweets(spark: SparkSession, path: String): DataFrame = {
    val schema = new StructType()
      .add("Id", LongType, false)
      .add("Tweet",StringType, false)
      .add("following",DoubleType, true)
      .add("followers",DoubleType, true)
      .add("actions", DoubleType, true)
      .add("is_retweet", DoubleType, true)
      .add("location", StringType, true)

    import spark.implicits._
    import org.apache.spark.sql.functions._

    spark.read.option("header", true)
      .schema(schema)
      .csv(path)
      .select($"Id",
      $"Tweet",
      $"following",
      $"followers",
      $"actions",
      $"is_retweet",
      $"location",
      when($"location".isNull, 0.0).otherwise(1.0).alias("has_location"))
      .where("Tweet is not null")
      .na.fill(0.0, Seq("following", "followers", "actions", "is_retweet"))
  }
}
