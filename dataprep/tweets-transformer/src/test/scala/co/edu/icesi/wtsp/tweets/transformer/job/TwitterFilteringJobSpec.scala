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

    val job = TwitterFilteringJob(spark,
      rawTweetsPath,
      output,
      Option(tfPipeline),
      Option("place is not null and lang = 'en'"))

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
  it should "process raw tweets without filtering" in {
    val output = testOutputPath + "/full_test"

    val job = TwitterFilteringJob(spark,
      rawTweetsPath,
      output)

    job.execute()

    Files.exists(Paths.get(s"$output/statistics/full_stats")) shouldBe false
    Files.exists(Paths.get(s"$output/statistics/condition_stats")) shouldBe false
    Files.exists(Paths.get(s"$output/statistics/final_stats")) shouldBe true
    Files.exists(Paths.get(s"$output/tweets")) shouldBe true

    //Load the results
    val results = spark.read.parquet(s"$output/tweets")

    //Check the statistics
    val finalStats = spark.read.json(s"$output/statistics/final_stats")
      .select("tweetCount", "userCount", "locationCount").head()

    finalStats.getLong(0) shouldBe 15585 //total tweets
    finalStats.getLong(1) shouldBe 10909 //total users
    finalStats.getLong(2) shouldBe 184 //total locations
    results.count() shouldBe finalStats.getLong(0)

    deleteRecursively(new File(testOutputPath))

  }
  it should "process raw tweets filtering only by lang is not null or undefined" in {
    val output = testOutputPath + "/full_test"

    val job = TwitterFilteringJob(spark,
      rawTweetsPath,
      output,
      filterExpression=Option("lang is not null and lang != 'und'"))

    job.execute()

    Files.exists(Paths.get(s"$output/statistics/full_stats")) shouldBe true
    Files.exists(Paths.get(s"$output/statistics/condition_stats")) shouldBe false
    Files.exists(Paths.get(s"$output/statistics/final_stats")) shouldBe true
    Files.exists(Paths.get(s"$output/tweets")) shouldBe true

    //Load the results
    val results = spark.read.parquet(s"$output/tweets")

    //Check the statistics
    val fullStats = spark.read.json(s"$output/statistics/full_stats")
      .select("tweetCount", "userCount", "locationCount").head()
    val finalStats = spark.read.json(s"$output/statistics/final_stats")
      .select("tweetCount", "userCount", "locationCount").head()

    fullStats.getLong(0) shouldBe 15585 //total tweets
    fullStats.getLong(1) shouldBe 10909 //total users
    fullStats.getLong(2) shouldBe 184 //total locations

    finalStats.getLong(0) shouldBe 10615 //total tweets
    finalStats.getLong(1) shouldBe 10549 //total users
    finalStats.getLong(2) shouldBe 178 //total locations
    results.count() shouldBe finalStats.getLong(0)

    deleteRecursively(new File(testOutputPath))
  }
  it should "process raw tweets filtering spam tweets only" in {
    val output = testOutputPath + "/full_test"

    val job = TwitterFilteringJob(spark,
      rawTweetsPath,
      output,
      spamPipelineModel=Option(tfPipeline))

    job.execute()

    Files.exists(Paths.get(s"$output/statistics/full_stats")) shouldBe true
    Files.exists(Paths.get(s"$output/statistics/condition_stats")) shouldBe false
    Files.exists(Paths.get(s"$output/statistics/final_stats")) shouldBe true
    Files.exists(Paths.get(s"$output/tweets")) shouldBe true

    //Load the results
    val results = spark.read.parquet(s"$output/tweets")

    //Check the statistics
    val fullStats = spark.read.json(s"$output/statistics/full_stats")
      .select("tweetCount", "userCount", "locationCount").head()
    val finalStats = spark.read.json(s"$output/statistics/final_stats")
      .select("tweetCount", "userCount", "locationCount").head()

    fullStats.getLong(0) shouldBe 15585 //total tweets
    fullStats.getLong(1) shouldBe 10909 //total users
    fullStats.getLong(2) shouldBe 184 //total locations

    finalStats.getLong(0) shouldBe 10574 //total tweets
    finalStats.getLong(1) shouldBe 10490 //total users
    finalStats.getLong(2) shouldBe 175 //total locations
    results.count() shouldBe finalStats.getLong(0)

    deleteRecursively(new File(testOutputPath))
  }
}
