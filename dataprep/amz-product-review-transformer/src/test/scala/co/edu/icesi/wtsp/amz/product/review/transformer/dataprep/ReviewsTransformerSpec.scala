package co.edu.icesi.wtsp.amz.product.review.transformer.dataprep

import co.edu.icesi.wtsp.amz.product.review.transformer.SpecCommon
import co.edu.icesi.wtsp.amz.product.review.transformer.products.reviews.ReviewsTransformer
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.AnalysisException
import org.scalatest.{FlatSpec, Matchers}
import org.apache.spark.sql.functions._

class ReviewsTransformerSpec extends FlatSpec
  with DataFrameSuiteBase
  with Matchers
  with SpecCommon{

  import spark.implicits._

  "The Reviews transformer" should "be able to read clean reviews" in {
    val expected = spark.read.parquet(transformedReviewsPath).orderBy($"summary")
    val productMetadata = spark.read.parquet(transformedMetadataPath)
    val reviews = spark.read.json(productReviewsPath)

    val reviewsTransformer = ReviewsTransformer(spark, productMetadata)

    val result = reviewsTransformer.transform(reviews)

    assertDataFrameEquals(expected, result)
  }
  it should "fail on invalid reviews" in {
    //Not all the columns available, corrupted input
    val input = spark.read.json(productReviewsPath).select($"asin", $"summary")
    val productMetadata = spark.read.parquet(transformedMetadataPath)

    val reviewsTransformer = ReviewsTransformer(spark, productMetadata)
    a[AnalysisException] should be thrownBy {
      reviewsTransformer.transform(input)
    }
  }
  it should "discard reviews with no review text" in {
    val productMetadata = spark.read.parquet(transformedMetadataPath)
    // We corrupt one records
    val (asin, title) = ("0028603540", "1,000 Lowfat Recipes (1,000 Recipes Series)")
    //put blank reviews on a specific record
    val reviews = spark.read.json(productReviewsPath)
      .select($"asin",
        $"summary",
        when($"asin" === asin, null).otherwise($"reviewText").as("reviewText"))

    val reviewsTransformer = ReviewsTransformer(spark, productMetadata)
    val result = reviewsTransformer.transform(reviews)

    //The result should not include the corrupt record
    result.where($"title" === title).count() shouldBe 0L

  }
  it should "discard reviews with too short review text" in {
    val productMetadata = spark.read.parquet(transformedMetadataPath)
    // We corrupt one records
    val (asin, title) = ("0028603540", "1,000 Lowfat Recipes (1,000 Recipes Series)")
    //put blank reviews on a specific record
    val reviews = spark.read.json(productReviewsPath)
      .select($"asin",
        $"summary",
        when($"asin" === asin, "Too short review").otherwise($"reviewText").as("reviewText"))

    val reviewsTransformer = ReviewsTransformer(spark, productMetadata)
    val result = reviewsTransformer.transform(reviews)

    //The result should not include the corrupt record
    result.where($"title" === title).count() shouldBe 0L
  }
  it should "apply the limit accordingly" in {
    val productMetadata = spark.read.parquet(transformedMetadataPath)
    val reviews = spark.read.json(productReviewsPath)

    val limit = Some(1000)
    val reviewsTransformer = ReviewsTransformer(spark, productMetadata, limit)

    val result = reviewsTransformer.transform(reviews)

    val initialRecords = reviews.count()
    val threshold = 0.1
    val resultCount = result.count().intValue()
    val lowerBound = limit.get - (initialRecords * threshold)
    val upperBound = limit.get + (initialRecords * threshold)

    resultCount should be >= lowerBound.intValue()
    resultCount should be <= upperBound.intValue()

  }
  it should "apply the limit accordingly with seed" in {
    val productMetadata = spark.read.parquet(transformedMetadataPath)
    val reviews = spark.read.json(productReviewsPath)

    val limit = Some(1000)
    val seed = Some(123)
    val reviewsTransformer = ReviewsTransformer(spark, productMetadata, limit, seed)

    val result = reviewsTransformer.transform(reviews)

    val initialRecords = reviews.count()
    val threshold = 0.1
    val resultCount = result.count().intValue()
    val lowerBound = limit.get - (initialRecords * threshold)
    val upperBound = limit.get + (initialRecords * threshold)

    resultCount should be >= lowerBound.intValue()
    resultCount should be <= upperBound.intValue()

    //if we execute again we should get exactly the same records
    val result2 = reviewsTransformer.transform(reviews)

    assertDataFrameEquals(result, result2)
  }

}
