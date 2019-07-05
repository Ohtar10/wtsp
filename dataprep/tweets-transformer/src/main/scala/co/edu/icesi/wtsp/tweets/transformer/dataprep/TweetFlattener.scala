package co.edu.icesi.wtsp.tweets.transformer.dataprep

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Column, DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

trait TransformerBuilder {
 type  Builder <: TransformerBuilder
}

trait TweetFlattenerAttributes extends TransformerBuilder {
  def withCols(cols: Column*): Builder
  def withFilter(filter: String): Builder
}

case class TweetFlattenerBuilder(
   cols: Option[Seq[Column]] = None,
   filter: Option[String] = None
   ) extends TweetFlattenerAttributes{

  override type Builder = TweetFlattenerBuilder

  override def withCols(cols: Column*): Builder = this.copy(cols = Option(cols))
  override def withFilter(filter: String): Builder = this.copy(filter = Option(filter))

  def withCol(col: Column): Builder = {
    val newCols = cols.getOrElse(Seq[Column]()) :+ col
    this.copy(cols = Option(newCols))
  }

  def build: TweetFlattener = {
    new TweetFlattener(cols.get, filter.get)
  }
}

class TweetFlattener(
                    cols: Seq[Column],
                    filter: String
                    ) extends Transformer{

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def copy(extra: ParamMap): Transformer = ???

  override def transformSchema(schema: StructType): StructType = ???

  override val uid: String = Identifiable.randomUID("tweet-flattener")

}
