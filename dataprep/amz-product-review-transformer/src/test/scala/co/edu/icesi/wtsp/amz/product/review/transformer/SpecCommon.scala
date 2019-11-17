package co.edu.icesi.wtsp.amz.product.review.transformer

import java.io.File

trait SpecCommon {

  val resourcesBasePath = "src/test/resources"

  val productMetadataPath = s"$resourcesBasePath/metadata/product_metadata.json"
  val transformedMetadataPath = s"$resourcesBasePath/metadata/transformed_metadata"

  val productReviewsPath = s"$resourcesBasePath/reviews/product_reviews.json"
  val transformedReviewsPath = s"$resourcesBasePath/reviews/transformed_reviews"

  val categoryConfigPath = s"$resourcesBasePath/config/category_mapping.yml"

  val testOutputPath = s"$resourcesBasePath/testOutput/"

  /**
   * Deletes recursively the files specified
   * at the given File object.
   *
   * @param file the path to delete recursively
   */
  def deleteRecursively(file: File): Unit = {
    if (file.isDirectory)
    {
      file.listFiles().foreach(deleteRecursively)
    }
    if (file.exists && !file.delete)
    {
      throw new Exception(s"Unable to delete file: ${file.getAbsolutePath}")
    }
  }

}
