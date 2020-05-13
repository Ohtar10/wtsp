package co.edu.icesi.wtsp.amz.product.review.transformer.products.metadata

import co.edu.icesi.wtsp.amz.product.review.transformer.util.{CategoryParser, Common, JobLogging}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Column, DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._

class MetadataTransformer(val spark: SparkSession,
                          val categoryParser: CategoryParser)
  extends Transformer
  with JobLogging
  with Common {

  import spark.implicits._

  private var transformedMetadata: DataFrame = _

  override def transform(dataset: Dataset[_]): DataFrame = {
    logInfo(spark, "Transforming product metadata")
    val expandedCategories = expandCategories(dataset)
    transformedMetadata = transformCategories(expandedCategories)
    transformDocuments(transformedMetadata)
  }

  private def expandCategories(df: Dataset[_]): DataFrame = {
    logInfo(spark, "Expanding categories")
    df.select($"asin",
      explode(flatten($"categories")).as("category"),
      $"title",
      $"description")
  }

  private def generateCategoryColumn(): Column = {
    logInfo(spark, "Generating category map case column")
    val categoryMap = categoryParser.getCategoryMappings()
    val categories = categoryParser.getCategories()
    val firstCase = when($"category".isin(categoryMap(categories.head):_*), categories.head)
    categories.tail.foldLeft(firstCase){ (column, category) =>
      column.when($"category".isin(categoryMap(category):_*), category)
    }.otherwise("skip").as("category")
  }

  private def transformCategories(df: DataFrame): DataFrame = {
    logInfo(spark, "Transforming categories according to mapping")
    val categoryColumn = generateCategoryColumn()
    df.select($"asin",
      $"title",
      trim($"description").as("description"),
      categoryColumn)
      .filter($"category".isNotNull && $"category" =!= "skip")
      .distinct()
  }

  private def transformDocuments(df: DataFrame): DataFrame = {
    logInfo(spark, "Transforming product metadata into documents")
    df.filter($"title".isNotNull &&
      length(trim($"title")) >= 3 &&
      $"description".isNotNull &&
      length(trim($"description")) >= documentTextMinCharacters)
      .select($"category", concat_ws("\n", $"title", $"description").as("document"))
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