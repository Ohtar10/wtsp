package co.edu.icesi.wtsp.amz.product.review.transformer.job

import co.edu.icesi.wtsp.amz.product.review.transformer.products.metadata.MetadataTransformer
import co.edu.icesi.wtsp.amz.product.review.transformer.products.reviews.ReviewsTransformer
import co.edu.icesi.wtsp.amz.product.review.transformer.util.{CategoryParser, JobLogging}
import org.apache.spark.sql.SparkSession

class AmzProductReviewTransformerJob(spark: SparkSession,
                                     metadataInput: String,
                                     reviewsInput: String,
                                     output: String,
                                     categoryMappingFile: String)
  extends Job with JobLogging {

  override def execute(): Unit = {
    val metadataTransformer = MetadataTransformer(spark, CategoryParser(categoryMappingFile))
    val rawMetadata = spark.read.json(metadataInput)
    logInfo(spark, "Metadata Transformation")
    val productMetadata = metadataTransformer.transform(rawMetadata)

    val reviewTransformer = ReviewsTransformer(spark, productMetadata)
    val rawReviews = spark.read.json(reviewsInput)
    logInfo(spark, "Reviews Transformation")
    val productReviews = reviewTransformer.transform(rawReviews)

    logInfo(spark, "Persisting final result")
    productReviews.write.mode("overwrite").parquet(output)
  }
}

object AmzProductReviewTransformerJob{
  def apply(spark: SparkSession,
            metadataInput: String,
            reviewsInput: String,
            output: String,
            categoryMappingFile: String = CategoryParser.defaultMappingPath
            ): AmzProductReviewTransformerJob =

    new AmzProductReviewTransformerJob(spark,
      metadataInput,
      reviewsInput,
      output,
      categoryMappingFile)
}
