package co.edu.icesi.wtsp.amz.product.review.transformer.dataprep

import co.edu.icesi.wtsp.amz.product.review.transformer.SpecCommon
import co.edu.icesi.wtsp.amz.product.review.transformer.products.metadata.MetadataTransformer
import co.edu.icesi.wtsp.amz.product.review.transformer.util.CategoryParser
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.AnalysisException
import org.scalatest.{FlatSpec, Matchers}
import org.apache.spark.sql.functions._

class MetadataTransformerSpec extends FlatSpec
  with DataFrameSuiteBase
  with Matchers
  with SpecCommon{

  import spark.implicits._

  "The Metadata transformer" should "transform clean product metadata" in {

    val expected = spark.read.parquet(transformedMetadataPath)
    val input = spark.read.json(productMetadataPath)

    val metadataTransformer = MetadataTransformer(spark)
    val result = metadataTransformer.transform(input)

    assertDataFrameEquals(expected.sort($"asin"), result.sort($"asin"))

  }
  it should "merge similar categories" in {
    val productMetadata = spark.read.json(productMetadataPath)
    val categoryParser = CategoryParser(CategoryParser.defaultMappingPath)
    val categoryMap = categoryParser.getCategoryMappings()
    val clothesName = "Clothing, Shoes & Jewelry"
    //Select Clothes categories
    val clothes = categoryMap(clothesName)

    val expected = productMetadata.select($"asin",
      explode($"categories").as("categories"),
      $"title",
      $"description").
      select($"asin",
        explode($"categories").as("category"),
        $"description", $"title")
      .where($"category".isin(clothes:_*))
      .select($"asin").distinct()
      .count()

    val metadataTransformer = MetadataTransformer(spark)
    val result = metadataTransformer.transform(productMetadata)
      .select($"asin").distinct()
      .where(array_contains($"categories", clothesName)).count()

    result shouldBe expected

  }
  it should "fail on invalid input data" in {
    val input = spark.read.json(productMetadataPath)
    //we are going to invalidate some data to make the process fail
    val corruptedInput = input.select($"asin", $"title", $"description") //no categories
    val metadataTransformer = MetadataTransformer(spark)

    a[AnalysisException] should be thrownBy {
      metadataTransformer.transform(corruptedInput)
    }
  }
  it should "discard products without category" in {
    val productMetadata = spark.read.json(productMetadataPath)


    val input = productMetadata.select($"asin",
      when($"asin" === "0000032034", null).otherwise($"categories").as("categories"),
      $"description", $"title")

    val metadataTransformer = MetadataTransformer(spark)
    val result = metadataTransformer.transform(input)

    val bookCount = result.where($"asin" === "0000032034").count().toInt

    result.count().toInt should be > 0
    bookCount shouldBe 0

  }

}
