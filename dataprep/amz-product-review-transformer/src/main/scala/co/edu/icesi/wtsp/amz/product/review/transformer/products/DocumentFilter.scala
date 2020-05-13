package co.edu.icesi.wtsp.amz.product.review.transformer.products

import co.edu.icesi.wtsp.amz.product.review.transformer.util.{CategoryParser, Common, JobLogging}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

class DocumentFilter(spark: SparkSession,
                     categoryParser: CategoryParser,
                     metadataCols: Seq[String],
                     reviewsCols: Seq[String]) extends JobLogging
  with Common {

  import spark.implicits._

  /**
   * Filter Products.
   *
   * Filter products in the provided metadata
   * and reviews data frames, according to the
   * provided category list.
   *
   * @param metadata data frame
   * @param reviews data frame
   * @return
   */
  def filterProductsIn(metadata: DataFrame,
                       reviews: DataFrame): (DataFrame, DataFrame) = {
    val categories = categoryParser.getCategoryMappings().values.flatMap(_.toList).toSeq
    val metadataSelect = metadataCols.map(col)
    val reviewsSelect = reviewsCols.map(col)

    val filteredMetadata = metadata.filter(arrays_overlap(flatten($"categories"), typedLit(categories)))
      .select(metadataSelect:_*).cache()
    val filteredReviews = reviews.as("rev").join(
      filteredMetadata.as("prod"), $"rev.asin" === $"prod.asin")
      .select($"rev.*").select(reviewsSelect:_*)
    (filteredMetadata, filteredReviews)
  }

}

object DocumentFilter {
  def apply(spark: SparkSession,
            categoryParser: CategoryParser,
            metadataCols: Seq[String],
            reviewsCols: Seq[String]): DocumentFilter =
    new DocumentFilter(spark,
      categoryParser,
      metadataCols,
      reviewsCols)
}
