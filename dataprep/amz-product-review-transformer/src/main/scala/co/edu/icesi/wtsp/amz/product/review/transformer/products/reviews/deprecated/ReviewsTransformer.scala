package co.edu.icesi.wtsp.amz.product.review.transformer.products.reviews.deprecated

import co.edu.icesi.wtsp.amz.product.review.transformer.util.{Common, JobLogging}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

@deprecated("Deprecated, use the transformer in the parent package")
class ReviewsTransformer(spark: SparkSession,
                         productMetadata: DataFrame,
                         limit: Option[Int],
                         sampleSeed: Option[Int]) extends Transformer
  with JobLogging
  with Common{

  import spark.implicits._

  override def transform(dataset: Dataset[_]): DataFrame = {
    logInfo(spark, "Transforming reviews")

    //obtain the reviews base from where to query the rest
    val base = limit match {
      case Some(n) =>
        val count = dataset.count()
        val toTake = if (count > n) n else count
        val fraction = 1.0 * toTake / count
        sampleSeed match {
          case Some(seed) => dataset.sample(fraction, seed).limit(n)
          case None => dataset.sample(fraction).limit(n)
        }

      case None => dataset
    }

    base.select($"asin", $"summary", $"reviewText".as("review_text"))
      .where($"review_text".isNotNull && (length($"review_text") >= documentTextMinCharacters))
      .join(productMetadata, "asin").
      select($"categories",
        trim(regexp_replace($"summary", textBlacklistRegex, "")).as("summary"),
        trim(regexp_replace($"review_text", textBlacklistRegex, "")).as("review_text"))
      .orderBy($"summary").cache()
  }

  override def copy(extra: ParamMap): Transformer = ???

  override def transformSchema(schema: StructType): StructType = ???

  override val uid: String = Identifiable.randomUID("product-reviews-transformer")
}

@deprecated("Deprecated, use the transformer in the parent package")
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
