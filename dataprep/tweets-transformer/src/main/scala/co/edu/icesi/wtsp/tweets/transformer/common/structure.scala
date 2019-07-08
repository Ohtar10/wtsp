package co.edu.icesi.wtsp.tweets.transformer.common

import org.apache.spark.ml.Transformer
import org.apache.spark.sql.Column

/**
  * Translator Builder trait
  */
trait TransformerBuilder{
  type Builder <: TransformerBuilder
  def build(): Transformer
}

/**
  * Transform Attributes trait.
  *
  * Contains the generic methods to specify
  * the spark sql columns and the filter expression
  * when transforming spark data frames.
  *
  */
trait TransformAttributes extends TransformerBuilder {
  def withCols(cols: Column*): Builder
  def withCol(col: Column): Builder
  def withFilter(filter: String): Builder
}
