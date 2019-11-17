package co.edu.icesi.wtsp.amz.product.review.transformer.products.reviews

import co.edu.icesi.wtsp.amz.product.review.transformer.util.{Common, JobLogging}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._

class ReviewsTransformer(spark: SparkSession, productMetadata: DataFrame) extends Transformer
  with JobLogging
  with Common{

  import spark.implicits._

  override def transform(dataset: Dataset[_]): DataFrame = {
    logInfo(spark, "Transforming reviews")
    dataset.select($"asin", $"summary", $"reviewText".as("review_text")).
      join(productMetadata, "asin").
      select($"categories",
        $"title",
        $"description",
        trim(regexp_replace($"summary", textBlacklistRegex, "")).as("summary"),
        trim(regexp_replace($"review_text", textBlacklistRegex, "")).as("review_text"))
      .where($"review_text".isNotNull && (length($"review_text") > reviewTextMinCharacters))
      .orderBy($"title")
  }

  override def copy(extra: ParamMap): Transformer = ???

  override def transformSchema(schema: StructType): StructType = ???

  override val uid: String = Identifiable.randomUID("product-reviews-transformer")
}

object ReviewsTransformer{
  def apply(spark: SparkSession, productMetadata: DataFrame): ReviewsTransformer =
    new ReviewsTransformer(spark, productMetadata)
}
