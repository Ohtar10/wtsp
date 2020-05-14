package co.edu.icesi.wtsp.amz.product.review.transformer.job

import co.edu.icesi.wtsp.amz.product.review.transformer.exceptions.InvalidStepArgumentException
import co.edu.icesi.wtsp.amz.product.review.transformer.products.DocumentFilter
import co.edu.icesi.wtsp.amz.product.review.transformer.products.metadata.MetadataTransformer
import co.edu.icesi.wtsp.amz.product.review.transformer.products.reviews.ReviewsTransformer
import co.edu.icesi.wtsp.amz.product.review.transformer.util.{CategoryParser, Common, JobLogging}
import org.apache.spark.sql.{DataFrame, SparkSession}

class ProductReviewDocumentTransformerJob(spark: SparkSession,
                                          parameters: Map[String, _],
                                          steps: Seq[String])
  extends Job with JobLogging with Common {

  private val outputParams: Set[String] = Set("full-documents-output",
  "review-documents-output",
  "metadata-documents-output",
  "filter-output")

  private def validate(): Unit = {
    if (!parameters.contains("metadata"))
      throw new InvalidStepArgumentException("The metadata is required.")
    if (!parameters.contains("reviews"))
      throw new InvalidStepArgumentException("The reviews are required.")
    if (parameters.keySet.intersect(outputParams).isEmpty)
      throw new InvalidStepArgumentException("No output configured, no data will be saved after processing.")
  }

  override def execute(): Unit = {
    logInfo(spark, "Executing pipeline Job")
    validate()
    val categoryParser = getCategoryParser
    val metadata = spark.read.json(getStringParam("metadata"))
    val reviews = spark.read.json(getStringParam("reviews"))
    val stepsI = steps.map(name => StepFactory.fromStepName(spark, name))

    var inputs = StepArgs(
      parameters
        + ("metadata" -> metadata)
        + ("reviews" -> reviews)
        + ("category-mappings" -> categoryParser)
    )

    //Execute all the provided steps
    stepsI.foreach{ step =>
      inputs = step.process(inputs)
      step.save(inputs)
    }
  }

  private def getCategoryParser: CategoryParser = {
    val categoryMapping = getAnyParam[Option[Either[String, String]]]("category-mappings")

    categoryMapping match {
      case Some(value) => value match{
        case Left(string) => CategoryParser.fromYamlString(string)
        case Right(string) => CategoryParser.fromYamlFile(string)
      }
      case None => throw new IllegalArgumentException("Invalid or no category-mappings specified")
    }
  }

  private def getStringParam(name: String): String = {
    parameters.getOrElse(name, "").toString
  }

  private def getAnyParam[T](name: String): T = {
    parameters.get(name).asInstanceOf[T]
  }

}

object ProductReviewDocumentTransformerJob {
  def apply(spark: SparkSession,
            parameters: Map[String, _],
            steps: Seq[String]): ProductReviewDocumentTransformerJob =
    new ProductReviewDocumentTransformerJob(spark,
                                        parameters,
                                        steps)
}

class FilterStep(spark: SparkSession) extends Step
  with JobLogging {

  override def validate(input: StepArgs): Unit = {
    if (!input.map.contains("metadata"))
      throw new InvalidStepArgumentException("The metadata is required.")
    if (!input.map.contains("reviews"))
      throw new InvalidStepArgumentException("The reviews are required.")
    if (!input.map.contains("metadata-cols"))
      throw new InvalidStepArgumentException("The metadata columns are required.")
    if (!input.map.contains("reviews-cols"))
      throw new InvalidStepArgumentException("The reviews columns are required.")
  }

  override def process(inputs: StepArgs): StepArgs = {
    logInfo(spark, s"Running step: ${this.getClass.getSimpleName}")
    validate(inputs)
    val metadata = inputs.map("metadata").asInstanceOf[DataFrame]
    val reviews = inputs.map("reviews").asInstanceOf[DataFrame]
    val metadataCols = inputs.map("metadata-cols").asInstanceOf[Seq[String]]
    val reviewsCols = inputs.map("reviews-cols").asInstanceOf[Seq[String]]
    val categoryParser = inputs.map("category-mappings").asInstanceOf[CategoryParser]
    val transformer = DocumentFilter(spark, categoryParser, metadataCols, reviewsCols)
    val (fMetadata, fReviews) = transformer.filterProductsIn(metadata, reviews)
    StepArgs(
      inputs.map
        + ("metadata" -> fMetadata)
        + ("reviews" -> fReviews)
    )
  }

  override def save(inputs: StepArgs): Unit = {
    val output = if (inputs.map.contains("filter-output")) Some(inputs.map("filter-output").toString) else None
    val categoryParser = inputs.map("category-mappings").asInstanceOf[CategoryParser]
    output match {
      case Some(path) if path.nonEmpty =>
        val metadata = inputs.map("metadata").asInstanceOf[DataFrame]
        val reviews = inputs.map("reviews").asInstanceOf[DataFrame]

        metadata.repartition(categoryParser.getCategories().size)
          .write
          .mode("overwrite")
          .parquet(s"$path/filtered-metadata")

        reviews.repartition(categoryParser.getCategories().size)
          .write
          .mode("overwrite")
          .parquet(s"$path/filtered-reviews")
      case None => //ignore if no output path provided
    }
  }
}

class MetadataTransformStep(spark: SparkSession) extends Step
  with JobLogging {

  override def validate(input: StepArgs): Unit = {
    if (!input.map.contains("metadata"))
      throw new InvalidStepArgumentException("The metadata is required.")
    if (!input.map.contains("reviews"))
      throw new InvalidStepArgumentException("The reviews are required.")
  }

  override def process(inputs: StepArgs): StepArgs = {
    logInfo(spark, s"Running step: ${this.getClass.getSimpleName}")
    validate(inputs)
    val metadata = inputs.map("metadata").asInstanceOf[DataFrame]
    val categoryParser = inputs.map("category-mappings").asInstanceOf[CategoryParser]
    val transformer = MetadataTransformer(spark, categoryParser)
    val reviewsDocuments = transformer.transform(metadata)
    StepArgs(
      inputs.map
      + ("metadata" -> reviewsDocuments)
      + ("transformed-metadata" -> transformer.getTransformedMetadata)
    )
  }

  override def save(inputs: StepArgs): Unit = {
    val output = if (inputs.map.contains("metadata-documents-output")) Some(inputs.map("metadata-documents-output").toString) else None
    val categoryParser = inputs.map("category-mappings").asInstanceOf[CategoryParser]
    output match {
      case Some(path) if path.nonEmpty =>
        logInfo(spark, s"Saving results in step: ${this.getClass.getSimpleName}")
        val documents = inputs.map("metadata").asInstanceOf[DataFrame]
        documents.repartition(categoryParser.getCategories().size)
          .write
          .mode("overwrite")
          .parquet(path)
      case None => //ignore if no output path provided
    }
  }
}

class ReviewTransformStep(spark: SparkSession) extends Step
  with JobLogging {

  override def validate(input: StepArgs): Unit = {
    if (!input.map.contains("metadata"))
      throw new InvalidStepArgumentException("The metadata is required.")
    if (!input.map.contains("reviews"))
      throw new InvalidStepArgumentException("The reviews are required.")
  }

  override def process(inputs: StepArgs): StepArgs = {
    logInfo(spark, s"Running step: ${this.getClass.getSimpleName}")
    validate(inputs)
    val metadata = inputs.map("transformed-metadata").asInstanceOf[DataFrame]
    val reviews = inputs.map("reviews").asInstanceOf[DataFrame]
    val limit = inputs.map.getOrElse("limit", None).asInstanceOf[Option[Int]]
    val seed = inputs.map.getOrElse("seed", None).asInstanceOf[Option[Int]]
    val transformer = ReviewsTransformer(spark, metadata, limit, seed)
    val reviewsDocuments = transformer.transform(reviews)
    StepArgs(
      inputs.map
      + ("reviews" -> reviewsDocuments)
    )
  }

  override def save(inputs: StepArgs): Unit = {
    val output = if (inputs.map.contains("review-documents-output")) Some(inputs.map("review-documents-output").toString) else None
    val categoryParser = inputs.map("category-mappings").asInstanceOf[CategoryParser]
    output match {
      case Some(path) if path.nonEmpty =>
        logInfo(spark, s"Saving results in step: ${this.getClass.getSimpleName}")
        val documents = inputs.map("reviews").asInstanceOf[DataFrame]
        documents.repartition(categoryParser.getCategories().size)
          .write
          .mode("overwrite")
          .parquet(path)
      case None => //ignore if no output path provided
    }
  }
}

class DocumentAggregatorStep(spark: SparkSession) extends Step
  with JobLogging {

  override def validate(input: StepArgs): Unit = {
    if (!input.map.contains("metadata"))
      throw new InvalidStepArgumentException("The metadata is required.")
    if (!input.map.contains("reviews"))
      throw new InvalidStepArgumentException("The reviews are required.")
  }

  override def process(inputs: StepArgs): StepArgs = {
    logInfo(spark, s"Running step: ${this.getClass.getSimpleName}")
    validate(inputs)
    val metadata = inputs.map("metadata").asInstanceOf[DataFrame]
    val reviews = inputs.map("reviews").asInstanceOf[DataFrame]
    val fullDocuments = metadata.union(reviews)
    StepArgs(
      inputs.map
      + ("documents" -> fullDocuments)
    )
  }

  override def save(inputs: StepArgs): Unit = {
    val output = if (inputs.map.contains("full-documents-output")) Some(inputs.map("full-documents-output").toString) else None
    val categoryParser = inputs.map("category-mappings").asInstanceOf[CategoryParser]
    output match {
      case Some(path) if path.nonEmpty =>
        logInfo(spark, s"Saving results in step: ${this.getClass.getSimpleName}")
        val documents = inputs.map("documents").asInstanceOf[DataFrame]
        documents.repartition(categoryParser.getCategories().size)
          .write
          .mode("overwrite")
          .parquet(path)
      case None => //ignore if no output path provided
    }
  }
}

object StepFactory {
  def fromStepName(spark: SparkSession, name: String): Step = {
    name match {
      case "filter" => new FilterStep(spark)
      case "transform-metadata" => new MetadataTransformStep(spark)
      case "transform-reviews" => new ReviewTransformStep(spark)
      case "aggregate-documents" => new DocumentAggregatorStep(spark)
    }
  }
}