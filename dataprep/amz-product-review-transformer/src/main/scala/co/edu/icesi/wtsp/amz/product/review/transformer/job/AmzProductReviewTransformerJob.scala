package co.edu.icesi.wtsp.amz.product.review.transformer.job

import co.edu.icesi.wtsp.amz.product.review.transformer.products.metadata.deprecated.MetadataTransformer
import co.edu.icesi.wtsp.amz.product.review.transformer.products.reviews.deprecated.ReviewsTransformer
import co.edu.icesi.wtsp.amz.product.review.transformer.util.{CategoryParser, Common, JobLogging}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

class AmzProductReviewTransformerJob(spark: SparkSession,
                                     metadataInput: String,
                                     reviewsInput: String,
                                     output: String,
                                     categoryMappingFile: String,
                                     limit: Option[Int],
                                     seed: Option[Int],
                                     strCat: Boolean)
  extends Job with JobLogging with Common{

  import spark.implicits._

  override def execute(): Unit = {
    val categoryParser = CategoryParser(categoryMappingFile)
    val metadataTransformer = MetadataTransformer(spark, categoryParser)
    val rawMetadata = spark.read.json(metadataInput)
    logInfo(spark, "Metadata Transformation")
    val productMetadata = metadataTransformer.transform(rawMetadata)

    val reviewTransformer = ReviewsTransformer(spark, productMetadata, limit, seed)
    val rawReviews = spark.read.json(reviewsInput)
    logInfo(spark, "Reviews Transformation")
    val productReviews = reviewTransformer.transform(rawReviews)

    logInfo(spark, "Persisting final result")
    val finalResult = createDocumentsDataFrame(productMetadata, productReviews)

    finalResult.repartition(categoryParser.getCategories().size).write
      .mode("overwrite")
      .parquet(output)
  }

  private def createDocumentsDataFrame(products: DataFrame, reviews: DataFrame): DataFrame = {

    val productsDocuments = products.select($"categories",
      concat(when($"title".isNotNull && length($"title") >= 3,
          concat($"title", lit("\n"))).otherwise(lit("")),
        when($"description".isNotNull && length($"description") >= 3,
          concat($"description", lit("\n"))).otherwise(lit(""))).as("document"))
      .where(length($"document") >= documentTextMinCharacters)

    val reviewsDocuments = reviews.select($"categories",
      concat(when($"summary".isNotNull && length($"summary") >= 3,
          concat($"summary", lit("\n"))).otherwise(lit("")),
        when($"review_text".isNotNull && length($"review_text") >= 3,
          concat($"review_text", lit("\n"))).otherwise(lit(""))).as("document"))
      .where(length($"document") >= documentTextMinCharacters)

    val fullDocuments = productsDocuments.union(reviewsDocuments)

    if (strCat) {
      fullDocuments.select(
        array_join($"categories", ";").as("categories"),
        $"document")
    } else {
      fullDocuments
    }
  }
}

object AmzProductReviewTransformerJob{
  def apply(spark: SparkSession,
            metadataInput: String,
            reviewsInput: String,
            output: String,
            categoryMappingFile: String = CategoryParser.defaultMappingPath,
            limit: Option[Int],
            seed: Option[Int],
            steps: Seq[String],
            strCat: Boolean = false
            ): AmzProductReviewTransformerJob =

    new AmzProductReviewTransformerJob(spark,
      metadataInput,
      reviewsInput,
      output,
      categoryMappingFile,
      limit,
      seed,
      strCat)
}
