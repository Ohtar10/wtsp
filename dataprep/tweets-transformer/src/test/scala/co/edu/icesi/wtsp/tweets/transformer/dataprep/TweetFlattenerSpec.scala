package co.edu.icesi.wtsp.tweets.transformer.dataprep

import co.edu.icesi.wtsp.tweets.transformer.SpecCommon
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.mockito.MockitoSugar
import org.scalatest.{FlatSpec, Matchers}

class TweetFlattenerSpec extends FlatSpec
  with MockitoSugar
  with Matchers
  with DataFrameSuiteBase
  with SpecCommon {

  "The tweet flattener" should "transform raw tweets according to given fields" in {
    val rawTweets = spark.read.json(rawTweetsPath)

    val flattenedTweets = TweetFlanner()
      .withFields()
      .flatten(rawTweets)

    flattenedTweets.count() shouldBe rawTweets.count()

  }

}
