package co.edu.icesi.wtsp.amz.product.review.transformer.dataprep

import co.edu.icesi.wtsp.amz.product.review.transformer.SpecCommon
import co.edu.icesi.wtsp.amz.product.review.transformer.products.DocumentFilter
import co.edu.icesi.wtsp.amz.product.review.transformer.util.CategoryParser
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.{FlatSpec, Matchers}
import org.apache.spark.sql.functions._

class DocumentFilterSpec extends FlatSpec
  with DataFrameSuiteBase
  with Matchers
  with SpecCommon{

  import spark.implicits._

  val metadataCols = Seq("asin", "title", "description", "categories")
  val reviewsCols = Seq("asin", "summary", "reviewText")

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

  val categoryParser: CategoryParser = CategoryParser.fromYamlString(categoryMappingsYaml)

  "The reviews Filter" should "filter out product categories not in list" in {
    val metadata = spark.read.json(productMetadataPath)
    val reviews = spark.read.json(productReviewsPath)
    val expectedMetadata = spark.read.json(filteredProductMetadataPath)
    val expectedReviews = spark.read.json(filteredReviewsPath)

    val documentFilterer = DocumentFilter(spark, categoryParser, metadataCols, reviewsCols)
    val (resMetadata, resReviews) = documentFilterer.filterProductsIn(metadata, reviews)

    val metadataSelect = metadataCols.map(col)
    val reviewsSelect = reviewsCols.map(col)
    assertDataFrameEquals(resMetadata.select(metadataSelect:_*).sort($"asin"),
      expectedMetadata.select(metadataSelect:_*).sort($"asin"))
    assertDataFrameEquals(resReviews.select(reviewsSelect:_*).sort($"asin"),
      expectedReviews.select(reviewsSelect:_*).sort($"asin"))
  }

}
