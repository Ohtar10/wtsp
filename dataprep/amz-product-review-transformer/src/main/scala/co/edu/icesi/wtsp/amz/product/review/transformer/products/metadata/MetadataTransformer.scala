package co.edu.icesi.wtsp.amz.product.review.transformer.products.metadata

import co.edu.icesi.wtsp.amz.product.review.transformer.util.{CategoryParser, Common, JobLogging, TransformCategoryUDF}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

class MetadataTransformer(val spark: SparkSession,
                          val categoryParser: CategoryParser)
  extends Transformer
  with JobLogging
  with Common {

  import spark.implicits._

  private var transformedMetadata: DataFrame = _

  override def transform(dataset: Dataset[_]): DataFrame = {
    logInfo(spark, "Transforming product metadata")
    val flattenedCategories = flattenCategories(dataset)
    transformedMetadata = transformCategories(flattenedCategories)
    transformDocuments(transformedMetadata)
  }

  private def flattenCategories(df: Dataset[_]): DataFrame = {
    logInfo(spark, "Expanding categories")
    df.select($"asin",
      flatten($"categories").as("categories"),
      $"title",
      $"description")
  }

  private def transformCategories(df: DataFrame): DataFrame = {
    logInfo(spark, "Transforming categories according to mapping")
    val transform_categories = TransformCategoryUDF.build(categoryParser)
    df.select($"asin",
      $"title",
      trim($"description").as("description"),
      transform_categories($"categories").as("categories")
    ).filter($"categories".isNotNull && length($"categories") > 0)
      .distinct()
  }

  private def transformDocuments(df: DataFrame): DataFrame = {
    logInfo(spark, "Transforming product metadata into documents")
    df.filter($"title".isNotNull &&
      length(trim($"title")) >= 3 &&
      $"description".isNotNull &&
      length(trim($"description")) >= documentTextMinCharacters)
      .select($"categories", concat_ws("\n", $"title", $"description").as("document"))
  }

  def getTransformedMetadata: DataFrame = transformedMetadata

  override def copy(extra: ParamMap): Transformer = ???

  override def transformSchema(schema: StructType): StructType = ???

  override val uid: String = Identifiable.randomUID("product-metadata-transformer")
}

object MetadataTransformer {
  def apply(spark: SparkSession,
            categoryParser: CategoryParser = CategoryParser.fromYamlFile(CategoryParser.defaultMappingPath)): MetadataTransformer =
    new MetadataTransformer(spark, categoryParser)
}