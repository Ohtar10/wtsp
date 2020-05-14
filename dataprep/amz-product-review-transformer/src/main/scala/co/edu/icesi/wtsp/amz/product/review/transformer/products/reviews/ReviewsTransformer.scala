package co.edu.icesi.wtsp.amz.product.review.transformer.products.reviews

import co.edu.icesi.wtsp.amz.product.review.transformer.util.{Common, JobLogging}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

class ReviewsTransformer(spark: SparkSession,
                         productMetadata: DataFrame,
                         limit: Option[Int],
                         sampleSeed: Option[Int]) extends Transformer
  with JobLogging
  with Common{

  import spark.implicits._

  override def transform(dataset: Dataset[_]): DataFrame = {
    logInfo(spark, "Transforming reviews")
    val base = limit match {
      case Some(n) =>
        val count = dataset.count()
        val toTake: Long = if (count > n) n else count
        val fraction = toTake.toDouble / count
        sampleSeed match {
          case Some(seed) => dataset.sample(fraction, seed)
          case None => dataset.sample(fraction)
        }
      case None => dataset
    }
    val filteredReviews = filterReviewsByProducts(base)
    transformToDocuments(filteredReviews)
  }

  private def filterReviewsByProducts(dataset: Dataset[_]): DataFrame = {
    logInfo(spark, "Filtering reviews by the surviving product categories")
    dataset.as("rev").join(productMetadata.as("prod"), $"rev.asin" === $"prod.asin")
      .select($"prod.category",
        trim(regexp_replace($"rev.summary", textBlacklistRegex, "")).as("summary"),
        trim(regexp_replace($"rev.reviewText", textBlacklistRegex, "")).as("review_text"))
  }

  private def transformToDocuments(df: DataFrame): DataFrame = {
    logInfo(spark, "Transforming reviews into product documents")
    df.filter($"summary".isNotNull
              && length(trim($"summary")) >= 3
              && $"review_text".isNotNull
              && length(trim($"review_text")) >= documentTextMinCharacters)
      .select($"category", concat_ws("\n", $"summary", $"review_text").as("document"))
  }

  override def copy(extra: ParamMap): Transformer = ???

  override def transformSchema(schema: StructType): StructType = ???

  override val uid: String = Identifiable.randomUID("product-reviews-transformer")
}

object ReviewsTransformer{
  def apply(spark: SparkSession,
            productMetadata: DataFrame,
            limit: Option[Int] = None,
            seed: Option[Int] = None): ReviewsTransformer =
    new ReviewsTransformer(spark,
      productMetadata,
      limit,
      seed)
}
