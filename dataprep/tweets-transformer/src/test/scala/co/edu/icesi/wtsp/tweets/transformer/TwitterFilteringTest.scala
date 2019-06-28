package co.edu.icesi.wtsp.tweets.transformer

import java.io.File

import co.edu.icesi.wtsp.util.UDF
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.FunSuite

class TwitterFilteringTest extends FunSuite with DataFrameSuiteBase{

  private val additionalFields = Set("created_timestamp", "year", "month", "day", "hour")

  test("filter tweets with location only"){
    val input: String = "src/test/resources/tweets/00"
    val output: String = "src/test/resources/output/"

    val tweetsDF = spark.read.json(input)
    val fields = tweetsDF.schema.fieldNames.toSet
    assert(tweetsDF.count() == 6562)

    val twitterFilter = TwitterFilter(spark, input, output)
    twitterFilter.filterWithExpression(expression = "coordinates is not null")

    val locTweetsDF = spark.read.parquet(output)
    //We are adding date related columns not present in the original dataset
    val filteredFields = locTweetsDF.schema.fieldNames.toSet -- additionalFields
    assert(locTweetsDF.count() == 83)
    assert(filteredFields.subsetOf(fields))

    //clean the results
    deleteRecursively(new File(output))
  }

  test("read tweets from multiple subdirectories") {
    val input: String = "src/test/resources/tweets/*/"
    val output: String = "src/test/resources/output"

    val tweetsDF = spark.read.json(input)
    val fields = tweetsDF.schema.fieldNames.toSet
    assert(tweetsDF.count() == 15585)

    val twitterFilter = TwitterFilter(spark, input, output)
    twitterFilter.filterWithExpression(expression = "coordinates is not null")

    val locTweetsDF = spark.read.parquet(output)
    val filtetedFields = locTweetsDF.schema.fieldNames.toSet -- additionalFields
    assert(locTweetsDF.count() == 205)
    assert(filtetedFields.subsetOf(fields))

    deleteRecursively(new File(output))
  }

  test("read tweets and select/flatten some fields"){
    val input: String = "src/test/resources/tweets/*/"
    val output: String = "src/test/resources/output"

    val tweetsDF = spark.read.json(input)
    assert(tweetsDF.count() == 15585)

    val twitterFilter = TwitterFilter(spark, input, output)
    val columns = Array(
      "id",
      "text",
      "lang",
      "created_timestamp",
      "year",
      "month",
      "day",
      "hour",
      "coordinates",
      "place"
    )
    twitterFilter.filterWithExpression(columns)

    val locTweetsDF = spark.read.parquet(output)
    locTweetsDF.printSchema()
    val filteredFields = locTweetsDF.schema.fieldNames

    assert(locTweetsDF.count() == 15585)
    assert(filteredFields.sameElements(columns))

    deleteRecursively(new File(output))
  }

  test("get wkt from json geometry"){
    val input: String = "src/test/resources/tweets/*/"
    val output: String = "src/test/resources/output"

    val tweetsDF = spark.read.json(input)
    assert(tweetsDF.count() == 15585)

    import org.apache.spark.sql.functions._
    import spark.implicits._

    val result = tweetsDF.select(to_json($"place.bounding_box").alias("bounding_box")).where("place is not null").collect()
    assert(result.length == 195)

    val wkt = result.map(r => {r.getAs[String]("bounding_box")}).map(s => UDF.wktFromGeoJson(s))
    assert(wkt.exists(_.isEmpty))
  }


  def deleteRecursively(file: File): Unit = {
    if (file.isDirectory)
    {
      file.listFiles().foreach(deleteRecursively)
    }
    if (file.exists && !file.delete)
    {
      throw new Exception(s"Unable to delete file: ${file.getAbsolutePath}")
    }
  }
}
