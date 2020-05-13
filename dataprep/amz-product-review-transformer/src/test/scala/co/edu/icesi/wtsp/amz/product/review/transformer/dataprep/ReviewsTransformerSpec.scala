package co.edu.icesi.wtsp.amz.product.review.transformer.dataprep

import co.edu.icesi.wtsp.amz.product.review.transformer.SpecCommon
import co.edu.icesi.wtsp.amz.product.review.transformer.products.metadata.MetadataTransformer
import co.edu.icesi.wtsp.amz.product.review.transformer.products.reviews.ReviewsTransformer
import co.edu.icesi.wtsp.amz.product.review.transformer.schema.Schemas
import co.edu.icesi.wtsp.amz.product.review.transformer.util.CategoryParser
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.AnalysisException
import org.apache.spark.sql.functions._
import org.scalatest.{FlatSpec, Matchers}

class ReviewsTransformerSpec extends FlatSpec
  with DataFrameSuiteBase
  with Matchers
  with SpecCommon {

  import spark.implicits._

  val categoryParser: CategoryParser = CategoryParser.fromYamlString(categoryMappingsYaml)

  "The reviews transformer" should "be able to transform reviews into documents" in {
    val expected = spark.createDataFrame(spark.read.parquet(reviewsDocuments).rdd, Schemas.documentSchema)
    val reviews = spark.read.json(filteredReviewsPath)
    val metadata = spark.read.json(filteredProductMetadataPath)

    val metadataTransformer = MetadataTransformer(spark, categoryParser)
    metadataTransformer.transform(metadata)
    val metadataTransformed = metadataTransformer.getTransformedMetadata

    val reviewsTransformer = ReviewsTransformer(spark, metadataTransformed)
    val result = reviewsTransformer.transform(reviews)
    assertDataFrameEquals(expected, result)
  }
  it should "fail on invalid reviews" in {
    //Not all the columns available, corrupted input
    val input = spark.read.json(filteredReviewsPath).select($"asin", $"summary")
    val metadata = spark.read.json(filteredProductMetadataPath)

    val metadataTransformer = MetadataTransformer(spark, categoryParser)
    metadataTransformer.transform(metadata)
    val productMetadata = metadataTransformer.getTransformedMetadata

    val reviewsTransformer = ReviewsTransformer(spark, productMetadata)
    a[AnalysisException] should be thrownBy {
      reviewsTransformer.transform(input)
    }
  }
  it should "discard reviews with no review text" in {
    val metadata = spark.read.json(filteredProductMetadataPath)

    val metadataTransformer = MetadataTransformer(spark, categoryParser)
    metadataTransformer.transform(metadata)
    val productMetadata = metadataTransformer.getTransformedMetadata

    //corrupt one record
    val reviews = spark.read.json(filteredReviewsPath)
      .select($"asin",
             $"summary",
             when($"asin" === "0005119367", null)
               .otherwise($"reviewText").as("reviewText"))

    val reviewsTransformer = ReviewsTransformer(spark, productMetadata)
    val result = reviewsTransformer.transform(reviews)

    result.filter($"document".contains("Awesome movie and great story. Highly recommend")).count() shouldBe 0
  }
  it should "discard reviews with too short review text" in {
    val metadata = spark.read.json(filteredProductMetadataPath)

    val metadataTransformer = MetadataTransformer(spark, categoryParser)
    metadataTransformer.transform(metadata)
    val productMetadata = metadataTransformer.getTransformedMetadata

    //corrupt one record
    val reviews = spark.read.json(filteredReviewsPath)
      .select($"asin",
        $"summary",
        when($"asin" === "0005119367", "Awesome movie and great story. Highly recommend")
          .otherwise($"reviewText").as("reviewText"))

    val reviewsTransformer = ReviewsTransformer(spark, productMetadata)
    val result = reviewsTransformer.transform(reviews)

    result.filter($"document".contains("Awesome movie and great story. Highly recommend")).count() shouldBe 0
  }
  it should "apply limit accordingly" in {
    val reviews = spark.read.json(filteredReviewsPath)
    val metadata = spark.read.json(filteredProductMetadataPath)

    val metadataTransformer = MetadataTransformer(spark, categoryParser)
    metadataTransformer.transform(metadata)
    val metadataTransformed = metadataTransformer.getTransformedMetadata

    val limit = Some(10)
    val reviewsTransformer = ReviewsTransformer(spark, metadataTransformed, limit = limit)
    val result = reviewsTransformer.transform(reviews)

    val initialRecords = reviews.count()
    val threshold = 0.1
    val resultCount = result.count().intValue()
    val lowerBound = limit.get - (initialRecords * threshold)
    val upperBound = limit.get + (initialRecords * threshold)

    resultCount should be >= lowerBound.intValue()
    resultCount should be <= upperBound.intValue()
  }
  it should "apply limit with seed accordingly" in {
    val reviews = spark.read.json(filteredReviewsPath)
    val metadata = spark.read.json(filteredProductMetadataPath)

    val metadataTransformer = MetadataTransformer(spark, categoryParser)
    metadataTransformer.transform(metadata)
    val metadataTransformed = metadataTransformer.getTransformedMetadata

    val limit = Some(10)
    val seed = Some(1234)
    val reviewsTransformer = ReviewsTransformer(spark, metadataTransformed, limit = limit, seed = seed)
    val result = reviewsTransformer.transform(reviews)

    val initialRecords = reviews.count()
    val threshold = 0.1
    val resultCount = result.count().intValue()
    val lowerBound = limit.get - (initialRecords * threshold)
    val upperBound = limit.get + (initialRecords * threshold)

    resultCount should be >= lowerBound.intValue()
    resultCount should be <= upperBound.intValue()

    //If we execute again we should get exactly the same records
    val result2 = reviewsTransformer.transform(reviews)
    assertDataFrameEquals(result, result2)
  }
}
