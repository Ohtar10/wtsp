package co.edu.icesi.wtsp.tweets.transformer.spamfilter

import org.apache.spark.ml.{PipelineModel, Transformer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Column, DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._
/**
  * Tweet Spam Assassin Pipeline
  *
  * This class loads and creates a pipeline to transform
  * raw tweets, i.e., using the default Twitter API JSON schema,
  * into the suitable input for the pipeline finally appending
  * a column determining if the tweet is spam (spam = 1) or ham (spam = 0).
  *
  * This class assumes the input data frame complies with Twitter API JSON schema
  *
  * @param pipeline the spam pipeline to apply
  */
class TweetSpamAssassinPipeline(val pipeline: PipelineModel) extends Transformer{

  val fromFields: Array[Column] = Array(
    new Column("id").alias("Id"),
    new Column("text").alias("Tweet"),
    new Column("user.followers_count").alias("followers"),
    new Column("user.friends_count").alias("following"),
    coalesce(new Column("retweet_count"), lit(0))
      .+(coalesce(new Column("favorite_count"), lit(0)))
      .+(coalesce(new Column("reply_count"), lit(0)))
      .alias("actions"),
    new Column("user.location").alias("location"),
    when(new Column("location").isNull, 0.0)
      .otherwise(1.0)
      .alias("has_location")
  )

  override def transform(dataset: Dataset[_]): DataFrame = {
    val originalColumns = dataset.columns

    val predictions = pipeline.transform(
      dataset.select(fromFields:_*)
    )
    dataset.join(predictions, dataset("id") === predictions("Id"))
  }

  override def copy(extra: ParamMap): Transformer = ???

  override def transformSchema(schema: StructType): StructType = ???

  override val uid: String = Identifiable.randomUID("tweet-spam-assassin")
}

object TweetSpamAssassinPipeline {

  def apply(pipelinePath: String): TweetSpamAssassinPipeline = {
    val pipelineModel = PipelineModel.load(pipelinePath)
    new TweetSpamAssassinPipeline(pipelineModel)
  }

}
