package co.edu.icesi.wtsp.amz.product.review.transformer

import co.edu.icesi.wtsp.amz.product.review.transformer.job.AmzProductReviewTransformerJob
import org.apache.spark.sql.SparkSession
import scopt.OptionParser

private object Metadata {
  val version: String = "0.1.0"
}

private object AppUtils {
  val parser = new OptionParser[Config]("amz-product-review-transformer") {
    head("amz-product-review-transformer", Metadata.version)

    opt[String]('m', "metadata").required().valueName("<metadata-path>")
      .action((x, c)=> c.copy(metadataInput = x))
      .text("The input path of the JSON metadata is required.")

    opt[String]('r', "reviews").required().valueName("<reviews-path>")
      .action((x, c)=> c.copy(reviewsInput = x))
      .text("The input path of the JSON reviews is required.")

    opt[String]('o', "output").required().valueName("<output-path>")
      .action((x, c)=> c.copy(output = x))
      .text("The output path to store the results is required.")

    opt[String]('c', "category-maps").required().valueName("<category-mappings>")
        .action((x, c)=> c.copy(categoryMappingFile = x))
        .text("The category mapping file is required.")

    opt[Int]('l', "limit").valueName("<record-limit>")
        .action((x, c) => c.copy(limit = x))
        .text("If you want to limit the amount of records obtained from the process")

    opt[Int]('s', "seed").valueName("<sample-seed>")
        .action((x, c) => c.copy(seed = x))
        .text("The random seed for the limit option if you want reproducible results")

    help('h', "help").text("Prints this usage text")
  }

  def createSparkSession(name: String = "Amazon Product Review Transformer App",
                         master: String = "local[*]",
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
                 metadataInput: String = "",
                 reviewsInput: String = "",
                 output: String = "",
                 categoryMappingFile: String = "",
                 limit: Int = 0,
                 seed: Int = 0
                 )

/**
  * Use this to test the app locally, from sbt:
  * sbt "run inputFile.txt outputFile.txt"
  *  (+ select TwitterFilteringLocalApp when prompted)
  */
object AmzProductReviewTransformerLocalApp extends App {

  AppUtils.parser.parse(args, Config()) match {
    case Some(config) => {
      val spark = AppUtils.createSparkSession()
      Runner.run(spark,
        config.metadataInput,
        config.reviewsInput,
        config.output,
        config.categoryMappingFile,
        config.limit,
        config.seed)
    }
    case _ => //Bad arguments. A message should have been displayed.
  }

}

/**
  * Use this when submitting the app to a cluster with spark-submit
  * */
object AmzProductReviewTransformerApp extends App{

  AppUtils.parser.parse(args, Config()) match {
    case Some(config) => {
      val spark = AppUtils.createSparkSession(hiveSupport = true)
      Runner.run(spark,
        config.metadataInput,
        config.reviewsInput,
        config.output,
        config.categoryMappingFile,
        config.limit,
        config.seed)
    }
    case _ => //Bad arguments. A message should have been displayed.
  }

}

object Runner {
  def run(spark: SparkSession,
          metadataInput: String,
          reviewsInput: String,
          output: String,
          categoryMappingFile: String,
          limit:Int,
          seed: Int): Unit = {

    AmzProductReviewTransformerJob(spark,
      metadataInput,
      reviewsInput,
      output,
      categoryMappingFile,
      if (limit > 0) Some(limit) else None,
      if (seed > 0) Some(seed) else None
    )
        .execute()
  }
}
