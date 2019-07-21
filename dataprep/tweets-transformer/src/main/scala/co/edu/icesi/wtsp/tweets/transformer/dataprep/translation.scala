package co.edu.icesi.wtsp.tweets.transformer.dataprep

import co.edu.icesi.wtsp.tweets.transformer.common.TransformAttributes
import co.edu.icesi.wtsp.tweets.transformer.schema.Schemas
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Column, DataFrame, Dataset}


/**
  * Tweet Transformer Builder
  *
  * This is a basic builder that takes a provided
  * list of spark columns and a filter and returns
  * and instance of the Tweet Transformer.
  *
  *
  * @param cols Spark columns to be used to transform
  * @param filter Spark sql where clause to filter when transforming
  */
case class TweetTransformerBuilder(
                                           cols: Option[Seq[Column]] = None,
                                           filter: Option[String] = None
                                         ) extends TransformAttributes{

  override type Builder = TweetTransformerBuilder

  /**
    * Adds the columns to the selection list.
    * @param cols spark columns
    * @return this
    */
  override def withCols(cols: Column*): Builder = this.copy(cols = Option(cols))

  /**
    * Adds the sql where clause to the transform operation.
    * @param filter spark sql where clause
    * @return this
    */
  override def withFilter(filter: String): Builder = this.copy(filter = Option(filter))

  /**
    * Adds a column to select from the passed data frame.
    *
    * @param col spark column type
    * @return this
    */
  override def withCol(col: Column): Builder = {
    val newCols = cols.getOrElse(Seq[Column]()) :+ col
    this.copy(cols = Option(newCols))
  }

  /**
    * Builds the transformer
    *
    * @return
    */
  override def build(): Transformer = {
    new TweetTransformer(
      cols = cols.getOrElse(Schemas.tweetObject),
      filter = filter.getOrElse(""))
  }
}

object TweetTransformerBuilder{
  def apply(): TweetTransformerBuilder = new TweetTransformerBuilder()
}

/**
  * Tweet Transformer
  *
  * This class will apply the column transformations to the
  * provided data frame. Also, it will apply the provided filter
  * if any.
  * @param cols spark columns
  * @param filter spark sql where clause
  */
class TweetTransformer(
                    cols: Seq[Column],
                    filter: String
                    ) extends Transformer
                      with Logging{

  override def transform(dataset: Dataset[_]): DataFrame = {
    logInfo(s"Transforming tweets with columns: $cols and filter expression: $filter")
    if (filter.nonEmpty)
      dataset.select(cols:_*).where(filter)
    else
      dataset.select(cols:_*)
  }

  override def copy(extra: ParamMap): Transformer = ???

  override def transformSchema(schema: StructType): StructType = ???

  override val uid: String = Identifiable.randomUID("tweet-transformer")

}
