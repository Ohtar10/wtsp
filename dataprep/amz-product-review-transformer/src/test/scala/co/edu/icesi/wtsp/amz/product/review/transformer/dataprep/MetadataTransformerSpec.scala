package co.edu.icesi.wtsp.amz.product.review.transformer.dataprep

import co.edu.icesi.wtsp.amz.product.review.transformer.SpecCommon
import co.edu.icesi.wtsp.amz.product.review.transformer.products.metadata.MetadataTransformer
import co.edu.icesi.wtsp.amz.product.review.transformer.schema.Schemas
import co.edu.icesi.wtsp.amz.product.review.transformer.util.CategoryParser
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.{AnalysisException, functions}
import org.apache.spark.sql.functions._
import org.scalatest.{FlatSpec, Matchers}

class MetadataTransformerSpec extends FlatSpec
  with DataFrameSuiteBase
  with Matchers
  with SpecCommon {

  import spark.implicits._

  val categoryParser: CategoryParser = CategoryParser.fromYamlString(categoryMappingsYaml)

  "The Metadata transformer" should "be able to transform product metadata" in {
    val expected = spark.createDataFrame(spark.read.parquet(metadataDocumentsPath).rdd, Schemas.documentSchema)
    val input = spark.read.json(filteredProductMetadataPath)

    val metadataTransformer = MetadataTransformer(spark, categoryParser)
    val result = metadataTransformer.transform(input)

    assertDataFrameEquals(expected.sort($"category"), result.sort($"category"))
  }
  it should "fail on invalid input data" in {
    val input = spark.read.json(filteredProductMetadataPath)
    //we are going to invalidate some data to make the process fail
    val corruptedInput = input.select($"asin", $"title", $"description") //no categories
    val metadataTransformer = MetadataTransformer(spark, categoryParser)

    a[AnalysisException] should be thrownBy {
      metadataTransformer.transform(corruptedInput)
    }
  }
  it should "discard products without category" in {
    val productMetadata = spark.read.json(filteredProductMetadataPath)
    val input = productMetadata.select($"asin",
      when($"asin" === "0000143561", null).otherwise($"categories").as("categories"),
      $"description", $"title")

    val metadataTransformer = MetadataTransformer(spark)
    val result = metadataTransformer.transform(input)

    val bookCount = result.where($"asin" === "0000143561").count().toInt

    result.count().toInt should be > 0
    bookCount shouldBe 0
  }
  it should "merge similar categories" in {
    val productMetadata = spark.read.json(filteredProductMetadataPath)
    val categoryMap = categoryParser.getCategoryMappings()
    val clothesName = "Clothing, Shoes & Jewelry"
    val clothes = categoryMap(clothesName)

    val metadataTransformer = MetadataTransformer(spark, categoryParser)

    val expected = productMetadata.select($"asin",
      trim($"title").as("title"),
      trim($"description").as("description"),
      explode(flatten($"categories")).as("category"))
      .filter($"category".isin(clothes:_*)
        && functions.length($"title") >= 3
        && functions.length($"description") >= metadataTransformer.documentTextMinCharacters)
      .select($"asin").distinct().count()

    val result = metadataTransformer.transform(productMetadata)
      .filter($"category" === clothesName).count()

    result shouldBe expected

  }

}
