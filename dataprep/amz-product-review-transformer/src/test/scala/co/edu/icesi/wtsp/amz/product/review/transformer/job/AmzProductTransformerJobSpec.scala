package co.edu.icesi.wtsp.amz.product.review.transformer.job

import java.io.File
import java.nio.file.{Files, Paths}

import co.edu.icesi.wtsp.amz.product.review.transformer.SpecCommon
import co.edu.icesi.wtsp.amz.product.review.transformer.util.CategoryParser
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.SparkException
import org.apache.spark.sql.AnalysisException
import org.scalatest.{FlatSpec, Matchers}

class AmzProductTransformerJobSpec extends FlatSpec
  with DataFrameSuiteBase
  with Matchers
  with SpecCommon{

  "The product review transformer job" should "be able to process products from a path" in {
    val expected = spark.read.parquet(transformedReviewsPath).orderBy("title")
    val job = AmzProductReviewTransformerJob(spark,
      productMetadataPath,
      productReviewsPath,
      testOutputPath,
      CategoryParser.defaultMappingPath,
      None,
      None)

    job.execute()

    Files.exists(Paths.get(testOutputPath)) shouldBe true
    val result = spark.read.parquet(testOutputPath).orderBy("title")

    assertDataFrameEquals(expected, result)

    deleteRecursively(new File(testOutputPath))
  }
  it should "fail if no proper paths are specified" in {
    val job = AmzProductReviewTransformerJob(spark,
      "invalid_metadata_path",
      productReviewsPath,
      testOutputPath,
      CategoryParser.defaultMappingPath,
      None,
      None)

    a[AnalysisException] should be thrownBy {
      job.execute()
    }

    val job2 = AmzProductReviewTransformerJob(spark,
      productMetadataPath,
      "invalid_product_reviews_path",
      testOutputPath,
      CategoryParser.defaultMappingPath,
      None,
      None)

    a[AnalysisException] should be thrownBy {
      job2.execute()
    }
  }
  it should "fail if invalid files are specified" in {
    //Here we pass the result as input, this should fail in later stages
    val job = AmzProductReviewTransformerJob(spark,
      transformedMetadataPath,
      productReviewsPath,
      testOutputPath,
      CategoryParser.defaultMappingPath,
      None,
      None)

    a[SparkException] should be thrownBy {
      job.execute()
    }
  }
  it should "run with custom category mapping" in {
    val job = AmzProductReviewTransformerJob(spark,
      productMetadataPath,
      productReviewsPath,
      testOutputPath,
      categoryConfigPath,
      None,
      None)

    job.execute()

    Files.exists(Paths.get(testOutputPath)) shouldBe true
    deleteRecursively(new File(testOutputPath))
  }

}
