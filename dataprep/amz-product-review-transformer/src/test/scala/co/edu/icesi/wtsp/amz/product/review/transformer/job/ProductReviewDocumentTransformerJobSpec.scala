package co.edu.icesi.wtsp.amz.product.review.transformer.job

import java.io.File
import java.nio.file.{Files, Paths}

import co.edu.icesi.wtsp.amz.product.review.transformer.SpecCommon
import co.edu.icesi.wtsp.amz.product.review.transformer.exceptions.{InvalidCategoryMappingException, InvalidStepArgumentException, StepExecutionException}
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.AnalysisException
import org.scalatest.{FlatSpec, Matchers}

class ProductReviewDocumentTransformerJobSpec extends FlatSpec
  with DataFrameSuiteBase
  with Matchers
  with SpecCommon {

  import spark.implicits._

  val metadataCols = Seq("asin", "title", "description", "categories")
  val reviewsCols = Seq("asin", "summary", "reviewText")
  val fullPipeline = Seq("filter", "transform-metadata", "transform-reviews", "aggregate-documents")

  "The product review document job (Full pipeline)" should "be able to generate documents from scratch" in {
    val expected = spark.read.parquet(fullDocumentsPath).orderBy($"document")
    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Left(categoryMappingsYaml),
        "full-documents-output" -> testOutputPath
      ),
      fullPipeline
      )

    job.execute()

    Files.exists(Paths.get(testOutputPath)) shouldBe true
    val result = spark.read.parquet(testOutputPath).orderBy($"document")

    assertDataFrameEquals(result, expected)
    deleteRecursively(new File(testOutputPath))
  }
  it should "fail if no proper input paths are specified" in {
    val job1 = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> "invalid path",
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Left(categoryMappingsYaml),
        "full-documents-output" -> testOutputPath
      ),
      fullPipeline
    )
    a[AnalysisException] should be thrownBy {
      job1.execute()
    }

    val job2 = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> "invalid path",
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Left(categoryMappingsYaml),
        "full-documents-output" -> testOutputPath
      ),
      fullPipeline
    )

    a[AnalysisException] should be thrownBy {
      job2.execute()
    }
  }
  it should "fail if invalid file content is provided" in {
    val job1 = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> fullDocumentsPath, //This is an invalid schema at this step
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Left(categoryMappingsYaml),
        "full-documents-output" -> testOutputPath
      ),
      fullPipeline
    )
    a[AnalysisException] should be thrownBy {
      job1.execute()
    }

    val job2 = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "reviews" -> fullDocumentsPath, //This is an invalid schema at this step
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Left(categoryMappingsYaml),
        "full-documents-output" -> testOutputPath
      ),
      fullPipeline
    )
    a[AnalysisException] should be thrownBy {
      job2.execute()
    }
  }
  it should "fail if invalid category mapping string is provided" in {
    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Left("invalid mapping"),
        "full-documents-output" -> testOutputPath
      ),
      fullPipeline
    )
    a[InvalidCategoryMappingException] should be thrownBy {
      job.execute()
    }
  }
  it should "fail if invalid category mapping file is provided" in {
    val invalidMapping = s"$resourcesBasePath/config/invalid_category_mapping.yml"
    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Right(invalidMapping),
        "full-documents-output" -> testOutputPath
      ),
      fullPipeline
    )
    a[InvalidCategoryMappingException] should be thrownBy {
      job.execute()
    }
  }
  it should "fail if no category mapping file is provided" in {
    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "full-documents-output" -> testOutputPath
      ),
      fullPipeline
    )
    a[IllegalArgumentException] should be thrownBy {
      job.execute()
    }
  }
  it should "run with a custom category mapping file" in {
    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Right(categoryConfigPath),
        "full-documents-output" -> testOutputPath
      ),
      fullPipeline
    )

    job.execute()

    Files.exists(Paths.get(testOutputPath)) shouldBe true
    deleteRecursively(new File(testOutputPath))
  }
  it should "work with output record limits" in {
    val limit = Some(20)
    val fullReviews = spark.read.parquet(reviewsDocuments).count()
    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "limit" -> limit,
        "seed" -> Some(1234),
        "category-mappings" -> Left(categoryMappingsYaml),
        "full-documents-output" -> testOutputPath
      ),
      fullPipeline
    )

    job.execute()

    Files.exists(Paths.get(testOutputPath)) shouldBe true
    val result = spark.read.parquet(testOutputPath).orderBy($"document").count()
    val threshold = 0.2
    val upperBound = limit.get + (fullReviews * threshold)
    val lowerBound = limit.get - (fullReviews * threshold)
    result.toInt should be >= lowerBound.intValue()
    result.toInt should be <= upperBound.intValue()

    deleteRecursively(new File(testOutputPath))
  }
  it should "fail if any output is configured" in {
    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Left(categoryMappingsYaml)
      ),
      fullPipeline
    )

    a[InvalidStepArgumentException] should be thrownBy {
      job.execute()
    }
  }

  "The product review document job (Document Filter Step)" should "be able to filter and persist the data" in {
    val expectedMetadata = s"$testOutputPath/filtered-metadata"
    val expectedReviews = s"$testOutputPath/filtered-reviews"

    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Left(categoryMappingsYaml),
        "filter-output" -> testOutputPath
      ),
      Seq("filter")
    )

    job.execute()

    Files.exists(Paths.get(expectedMetadata)) shouldBe true
    Files.exists(Paths.get(expectedReviews)) shouldBe true
    deleteRecursively(new File(testOutputPath))
  }
  it should "fail if no metadata columns are provided" in {
    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "reviews" -> productReviewsPath,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Left(categoryMappingsYaml)
      ),
      Seq("filter")
    )

    a[InvalidStepArgumentException] should be thrownBy {
      job.execute()
    }
  }
  it should "fail if no reviews columns are provided" in {
    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "category-mappings" -> Left(categoryMappingsYaml)
      ),
      Seq("filter")
    )

    a[InvalidStepArgumentException] should be thrownBy {
      job.execute()
    }
  }
  it should "fail if no metadata is provided" in {
    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "reviews" -> productReviewsPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Left(categoryMappingsYaml)
      ),
      Seq("filter")
    )

    a[InvalidStepArgumentException] should be thrownBy {
      job.execute()
    }
  }
  it should "fail if no reviews are provided" in {
    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> productMetadataPath,
        "metadata-cols" -> metadataCols,
        "reviews-cols" -> reviewsCols,
        "category-mappings" -> Left(categoryMappingsYaml)
      ),
      Seq("filter")
    )

    a[InvalidStepArgumentException] should be thrownBy {
      job.execute()
    }
  }

  "The product review document job (Metadata transform Step)" should "be able to transform and persist metadata documents" in {

    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> filteredProductMetadataPath,
        "reviews" -> filteredReviewsPath,
        "category-mappings" -> Left(categoryMappingsYaml),
        "metadata-documents-output" -> testOutputPath
      ),
      Seq("transform-metadata")
    )

    job.execute()

    Files.exists(Paths.get(testOutputPath)) shouldBe true
    deleteRecursively(new File(testOutputPath))
  }

  "The product review document job (Reviews transform Step)" should "be able to transform and persist review documents" in {

    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> filteredProductMetadataPath,
        "reviews" -> filteredReviewsPath,
        "category-mappings" -> Left(categoryMappingsYaml),
        "review-documents-output" -> testOutputPath
      ),
      Seq("transform-metadata", "transform-reviews")
    )

    job.execute()

    Files.exists(Paths.get(testOutputPath)) shouldBe true
    deleteRecursively(new File(testOutputPath))
  }

  "The product review document job (Document aggregator Step)" should "be able to aggregate and persist documents" in {

    val job = ProductReviewDocumentTransformerJob(spark,
      Map(
        "metadata" -> filteredProductMetadataPath,
        "reviews" -> filteredReviewsPath,
        "category-mappings" -> Left(categoryMappingsYaml),
        "full-documents-output" -> testOutputPath
      ),
      Seq("transform-metadata", "transform-reviews", "aggregate-documents")
    )

    job.execute()

    Files.exists(Paths.get(testOutputPath)) shouldBe true
    deleteRecursively(new File(testOutputPath))
  }




}
