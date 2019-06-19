package co.edu.icesi.wtsp.tweets.transformer

import java.io.File

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.FunSuite

class TwitterFilteringTest extends FunSuite with DataFrameSuiteBase{

  test("filter tweets with location only"){
    val input: String = "src/test/resources/tweets/00.json"
    val output: String = "src/test/resources/output/"

    val tweetsDF = spark.read.json(input)
    val fields = tweetsDF.schema.fieldNames.toSet
    assert(tweetsDF.count() == 3347)

    val twitterFilter = TwitterFilter(spark, input, output)
    twitterFilter.filterWithExpression("coordinates is not null")

    val locTweetsDF = spark.read.parquet(s"${output}")
    val filteredFields = locTweetsDF.schema.fieldNames.toSet
    assert(locTweetsDF.count() == 48)
    //assert(filteredFields.subsetOf(fields))

    //clean the results
    //deleteRecursively(new File(output))
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
