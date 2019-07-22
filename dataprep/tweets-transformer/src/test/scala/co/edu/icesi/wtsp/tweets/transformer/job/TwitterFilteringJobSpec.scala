package co.edu.icesi.wtsp.tweets.transformer.job

import java.io.File
import java.nio.file.{Files, Paths}

import co.edu.icesi.wtsp.tweets.transformer.SpecCommon
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.{FlatSpec, Matchers}

class TwitterFilteringJobSpec extends FlatSpec
  with DataFrameSuiteBase
  with Matchers
  with SpecCommon {

  "The twitter filtering job" should "process raw tweets and take only those with location" in {
    val output = testOutputPath + "/full_test"

    val job = TwitterFilteringJob(Option(spark), rawTweetsPath, output, tfPipeline)
    job.execute()

    Files.exists(Paths.get(s"$output/statistics/full_stats")) shouldBe true
    Files.exists(Paths.get(s"$output/statistics/condition_stats")) shouldBe true
    Files.exists(Paths.get(s"$output/statistics/final_stats")) shouldBe true
    Files.exists(Paths.get(s"$output/tweets")) shouldBe true

    //Load the results
    val results = spark.read.parquet(s"$output/tweets")

    //Check the statistics
    val fullStats = spark.read.json(s"$output/statistics/full_stats")
      .select("tweetCount", "userCount", "locationCount").head()
    val conditionStats = spark.read.json(s"$output/statistics/condition_stats")
      .select("tweetCount", "userCount", "locationCount").head()
    val finalStats = spark.read.json(s"$output/statistics/final_stats")
      .select("tweetCount", "userCount", "locationCount").head()


    fullStats.getLong(0) shouldBe 15585 //total tweets
    fullStats.getLong(1) shouldBe 10909 //total users
    fullStats.getLong(2) shouldBe 184 //total locations

    conditionStats.getLong(0) shouldBe 87 //total tweets
    conditionStats.getLong(1) shouldBe 86 //total users
    conditionStats.getLong(2) shouldBe 81 //total locations

    finalStats.getLong(0) shouldBe 87 //total tweets
    finalStats.getLong(1) shouldBe 86 //total users
    finalStats.getLong(2) shouldBe 81 //total locations
    results.count() shouldBe finalStats.getLong(0)

    deleteRecursively(new File(testOutputPath))
  }
}
