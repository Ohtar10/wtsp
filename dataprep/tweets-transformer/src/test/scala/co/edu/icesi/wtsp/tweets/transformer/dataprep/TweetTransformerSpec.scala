package co.edu.icesi.wtsp.tweets.transformer.dataprep

import co.edu.icesi.wtsp.tweets.transformer.SpecCommon
import co.edu.icesi.wtsp.tweets.transformer.schema.Schemas
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.types._
import org.mockito.MockitoSugar
import org.scalatest.{FlatSpec, Matchers}

class TweetTransformerSpec extends FlatSpec
  with MockitoSugar
  with Matchers
  with DataFrameSuiteBase
  with SpecCommon {

  val commonSchema = new StructType()
    .add("id", LongType, nullable = true)
    .add("tweet", StringType, nullable = true)
    .add("lang", StringType, nullable = true)
    .add("favorite_count", DoubleType, nullable = true)
    .add("retweet_count", DoubleType, nullable = true)
    .add("is_retweet", DoubleType, nullable = false)
    .add("user_id", LongType, nullable = true)
    .add("user_name", StringType, nullable = true)
    .add("user_followers_count", DoubleType, nullable = true)
    .add("user_following_count", DoubleType, nullable = true)
    .add("user_location", StringType, nullable = true)
    .add("created_timestamp", TimestampType, nullable = true)
    .add("year", IntegerType, nullable = true)
    .add("month", IntegerType, nullable = true)
    .add("day", IntegerType, nullable = true)
    .add("hour", IntegerType, nullable = true)
    .add("hashtags", StringType, nullable = false)
    .add("user_mentions", StringType, nullable = false)
    .add("user_id_mentions", StringType ,nullable = false)
    .add("expanded_urls", StringType ,nullable = false)
    .add("location_geometry", StringType ,nullable = true)
    .add("place_geometry", StringType ,nullable = true)
    .add("place_id", StringType ,nullable = true)
    .add("place_name", StringType ,nullable = true)
    .add("place_full_name", StringType ,nullable = true)
    .add("country", StringType ,nullable = true)
    .add("country_code", StringType ,nullable = true)
    .add("place_type", StringType ,nullable = true)
    .add("place_url", StringType ,nullable = true)

  "The tweet transformer" should "transform raw tweets into the common schema" in {
    val rawTweets = spark.read.schema(Schemas.sourceSchema).json(rawTweetsPath)

    val transformer = TweetTransformerBuilder()
      .withCols(Schemas.tweetObject:_*)
      .build()

    val transformedTweets = transformer.transform(rawTweets)
    transformedTweets.count() shouldBe rawTweets.count()
    transformedTweets.count() shouldBe 15585
    transformedTweets.schema shouldBe commonSchema
  }
  it should "transform raw tweets and apply a filter to get only desired rows" in {
    val rawTweets = spark.read.schema(Schemas.sourceSchema).json(rawTweetsPath)
    val condition = "place is not null"
    val filtered = rawTweets.select("*").where(condition)

    val transformer = TweetTransformerBuilder()
      .withCols(Schemas.tweetObject:_*)
      .withFilter(condition)
      .build()

    val transformedTweets = transformer.transform(rawTweets)
    transformedTweets.count() shouldBe filtered.count()
    transformedTweets.count() shouldBe 195
    transformedTweets.schema shouldBe commonSchema
  }

}
