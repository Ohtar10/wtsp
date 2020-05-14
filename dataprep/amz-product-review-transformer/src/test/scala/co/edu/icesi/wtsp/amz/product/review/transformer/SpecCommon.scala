package co.edu.icesi.wtsp.amz.product.review.transformer

import java.io.File

trait SpecCommon {

  val resourcesBasePath = "src/test/resources"

  val productMetadataPath = s"$resourcesBasePath/metadata/product_metadata.json"
  @deprecated("Deprecated, preserved for historic purposes, use metadataDocumentsPath instead")
  val transformedMetadataPath = s"$resourcesBasePath/metadata/transformed_metadata"
  val metadataDocumentsPath = s"$resourcesBasePath/metadata/metadata_documents/"

  val filteredProductMetadataPath = s"$resourcesBasePath/metadata/filtered_metadata/expected_filtered_metadata.json"
  val filteredReviewsPath = s"$resourcesBasePath/reviews/filtered_reviews/expected_filtered_reviews.json"

  val productReviewsPath = s"$resourcesBasePath/reviews/product_reviews.json"
  @deprecated("Deprecated, preserved for historic purposes, use reviewsDocuments instead")
  val transformedReviewsPath = s"$resourcesBasePath/reviews/transformed_reviews"
  val reviewsDocuments = s"$resourcesBasePath/reviews/reviews_documents"

  val finalDocumentsPath = s"$resourcesBasePath/documents"
  @deprecated("Deprecated, preserved for historic purposes")
  val documentsWithArrayCategoriesPath = s"$finalDocumentsPath/expected_docs.parquet"
  @deprecated("Deprecated, preserved for historic purposes")
  val documentsWithStringCategoriesPath = s"$finalDocumentsPath/expected_docs_str_cat.parquet"
  val fullDocumentsPath = s"$finalDocumentsPath/full_documents"

  val testOutputPath = s"$resourcesBasePath/testOutput/"

  val categoryConfigPath = s"$resourcesBasePath/config/category_mapping.yml"
  val categoryMappingsYaml =
    """
    categories:
      - name: "Movies & TV"
        mappings:
          - "Movies"
          - "Movies & TV"
      - name: "Clothing, Shoes & Jewelry"
        mappings:
          - "Clothing"
          - "T-Shirts"
          - "Shirts"
          - "Jewelry"
          - "Dresses"
          - "Boots"
          - "Shoes"
          - "Jewelry: International Shipping Available"
          - "Shoes & Accessories: International Shipping Available"
          - "Clothing, Shoes & Jewelry"
          - "Fashion"
          - "Earrings"
    """

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
