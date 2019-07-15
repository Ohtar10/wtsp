package co.edu.icesi.wtsp.tweets.transformer

import co.edu.icesi.wtsp.tweets.transformer.job.TwitterFilteringJob
import org.apache.spark.sql.SparkSession
import scopt.OptionParser

private object Metadata {
  val version: String = "1.0.0"
}

private object AppUtils {
  val parser = new OptionParser[Config]("tweets-transformer") {
    head("tweets-transformer", Metadata.version)

    opt[String]('i', "input").required().valueName("<input-path>")
      .action((x, c)=> c.copy(input = x))
      .text("The input path of the JSON tweets is required.")

    opt[String]('o', "output").required().valueName("<output-path>")
      .action((x, c)=> c.copy(output = x))
      .text("The output path to store the results is required.")

    opt[String]('s', "spam-pipeline").required().valueName("<spam-pipeline-path>")
      .action((x, c)=> c.copy(spamPipeline = x))
      .text("The path of the spam pipeline model is required.")

    opt[String]('f', "filter-expression").valueName("<sql-filter-expression>")
      .action((x, c) => c.copy(filterExpression = x))
      .text("SQL like filter expression according to the input schema.")

    help('h', "help").text("Prints this usage text")
  }

  def createSparkSession(name: String = "Twitter Filtering App",
                         master: String = "local",
                         hiveSupport: Boolean = false): SparkSession = {
    if (hiveSupport) {
      SparkSession.builder()
        .appName(name)
        .enableHiveSupport()
        .getOrCreate()
    } else {
      SparkSession.builder()
        .appName(name)
        .master(master)
        .getOrCreate()
    }
  }

}

case class Config(
                 input: String = "",
                 output: String = "",
                 spamPipeline: String = "",
                 filterExpression: String = ""
                 )

/**
  * Use this to test the app locally, from sbt:
  * sbt "run inputFile.txt outputFile.txt"
  *  (+ select TwitterFilteringLocalApp when prompted)
  */
object TwitterFilteringLocalApp extends App {

  AppUtils.parser.parse(args, Config()) match {
    case Some(config) => {
      val spark = AppUtils.createSparkSession()
      Runner.run(spark,
        config.input,
        config.output,
        config.spamPipeline,
        config.filterExpression)
    }
    case _ => //Bad arguments. A message should have been displayed.
  }

}

/**
  * Use this when submitting the app to a cluster with spark-submit
  * */
object TwitterFilteringApp extends App{

  AppUtils.parser.parse(args, Config()) match {
    case Some(config) => {
      val spark = AppUtils.createSparkSession(hiveSupport = true)
      Runner.run(spark,
        config.input,
        config.output,
        config.spamPipeline,
        config.filterExpression)
    }
    case _ => //Bad arguments. A message should have been displayed.
  }

}

object Runner {
  def run(spark: SparkSession,
          input: String,
          output: String,
          spamPipeline: String,
          expression: String): Unit = {

    TwitterFilteringJob(Option(spark),
      input,
      output,
      spamPipeline,
      expression)
        .execute()
  }
}
