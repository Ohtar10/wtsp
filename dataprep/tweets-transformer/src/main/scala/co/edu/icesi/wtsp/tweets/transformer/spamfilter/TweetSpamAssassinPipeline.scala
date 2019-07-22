package co.edu.icesi.wtsp.tweets.transformer.spamfilter

import co.edu.icesi.wtsp.tweets.transformer.schema.Schemas
import co.edu.icesi.wtsp.util.JobLogging
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{PipelineModel, Transformer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Column, DataFrame, Dataset}

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
class TweetSpamAssassinPipeline(val pipeline: PipelineModel) extends Transformer with JobLogging{

  val fromFields: Seq[Column] = Schemas.tweetSpamObject

  override def transform(dataset: Dataset[_]): DataFrame = {
    val context = "tweet-spam-assassin"
    logInfo(dataset.sparkSession, "Transforming and predicting spam vs ham tweets...", Option(context))
    val predictions = pipeline.transform(dataset.select(fromFields:_*))
      .withColumn("is_spam",
        when(new Column("prediction") === 1.0, 0.0).otherwise(1.0))
      .select("id", "is_spam")

    logInfo(dataset.sparkSession, "Preparing spam vs ham results...", Option(context))
    dataset.join(predictions, Seq("id"))
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
