package co.edu.icesi.wtsp.tweets.transformer

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
  * Use this to test the app locally, from sbt:
  * sbt "run inputFile.txt outputFile.txt"
  *  (+ select CountingLocalApp when prompted)
  */
object TwitterFilteringLocalApp extends App {
  val (inputFile, outputFile, fields, expression) = (args(0), args(1), args(2), args(3))
  val conf = new SparkConf()
    .setMaster("local")
    .setAppName("Twitter Filtering App")

  val spark = SparkSession.builder()
      .appName("Twitter Filtering App")
      .master("local")
      .getOrCreate()

  Runner.run(spark, inputFile, outputFile, fields, expression)
}

/**
  * Use this when submitting the app to a cluster with spark-submit
  * */
object TwitterFilteringApp extends App{
  val (inputFile, outputFile, fields, expression) = (args(0), args(1), args(2), args(3))

  // spark-submit command should supply all necessary config elements
  val spark = SparkSession.builder()
      .appName("Twitter Filtering App")
      .enableHiveSupport()
      .getOrCreate()
  Runner.run(spark, inputFile, outputFile, fields, expression)
}

object Runner {
  def run(spark: SparkSession, input: String, output: String, fields: String, expression: String): Unit = {
    TwitterFilter(spark, input, output).filterWithExpression(fields.split(","), expression)
  }
}
