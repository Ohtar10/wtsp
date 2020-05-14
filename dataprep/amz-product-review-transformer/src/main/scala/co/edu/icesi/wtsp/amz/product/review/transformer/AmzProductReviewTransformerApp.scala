package co.edu.icesi.wtsp.amz.product.review.transformer

import co.edu.icesi.wtsp.amz.product.review.transformer.job.{AmzProductReviewTransformerJob, ProductReviewDocumentTransformerJob}
import org.apache.spark.sql.SparkSession
import scopt.OptionParser

private object Metadata {
  val version: String = "0.1.0"
}

private object AppUtils {

  val parser: OptionParser[Config] = new OptionParser[Config]("amz-product-review-transformer") {
    head("amz-product-review-transformer", Metadata.version)
    val defaults: Config = Config()
    opt[String]('m', "metadata").required().valueName("<metadata-path>")
      .action((x, c)=> c.copy(metadataInput = x))
      .text("The input path of the JSON metadata is required.")

    opt[String]('r', "reviews").required().valueName("<reviews-path>")
      .action((x, c)=> c.copy(reviewsInput = x))
      .text("The input path of the JSON reviews is required.")

    opt[Map[String, String]]('o', "outputs").required().valueName("<k1=v1,k2=v2>")
      .action((x, c)=> c.copy(outputs = x))
      .text(s"The output paths to store the results is required. Possible keys: ${defaults.outputKeys.mkString(", ")}")

    opt[Seq[String]]('s', "steps").optional().valueName("<filter,transform-metadata,transform-reviews,aggregate-documents>")
        .action((x, c) => c.copy(steps = x))
        .text("The steps of the pipeline to execute, valid values: filter, transform-metadata, transform-reviews, aggregate-documents")

    opt[String]('c', "category-maps").required().valueName("<category-mappings>")
        .action((x, c)=> c.copy(categoryMappingFile = x))
        .text("The category mapping file is required.")

    opt[Int]('l', "limit").optional().valueName("<record-limit>")
        .action((x, c) => c.copy(limit = x))
        .text("If you want to limit the amount of records obtained from the process")

    opt[Int]("seed").optional().valueName("<sample-seed>")
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
                 outputs: Map[String, String] = Map.empty[String, String],
                 categoryMappingFile: String = "",
                 limit: Int = 0,
                 seed: Int = 0,
                 steps: Seq[String] = Seq("filter", "transform-metadata", "transform-reviews", "aggregate-documents")
                 ) {
  val outputKeys: Set[String] = Set("full-documents-output",
    "review-documents-output",
    "metadata-documents-output",
    "filter-output")
}

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
        config.outputs,
        config.categoryMappingFile,
        config.limit,
        config.seed,
        config.steps)
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
        config.outputs,
        config.categoryMappingFile,
        config.limit,
        config.seed,
        config.steps)
    }
    case _ => //Bad arguments. A message should have been displayed.
  }

}

object Runner {
  def run(spark: SparkSession,
          metadataInput: String,
          reviewsInput: String,
          output: Map[String, String],
          categoryMappingFile: String,
          limit: Int,
          seed: Int,
          steps: Seq[String]): Unit = {

    val limitOpt = if (limit > 0) Some(limit) else None
    val seedOpt = if (seed > 0) Some(seed) else None
    val parameters = output.asInstanceOf[Map[String, _]] +
      ("metadata" -> metadataInput) +
      ("reviews" -> reviewsInput) +
      ("category-mappings" -> Right(categoryMappingFile)) +
      ("limit" -> limitOpt) +
      ("seed" -> seedOpt)

    ProductReviewDocumentTransformerJob(spark,
      parameters,
      steps
      ).execute()
  }
}
