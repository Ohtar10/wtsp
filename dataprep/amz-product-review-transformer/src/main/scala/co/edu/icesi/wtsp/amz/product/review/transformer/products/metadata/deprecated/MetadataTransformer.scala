package co.edu.icesi.wtsp.amz.product.review.transformer.products.metadata.deprecated

import co.edu.icesi.wtsp.amz.product.review.transformer.util.{CategoryParser, Common, JobLogging}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

@deprecated("Deprecated, use the transformer in the parent package")
class MetadataTransformer(val spark: SparkSession,
                          val categoryParser: CategoryParser)
  extends Transformer
  with JobLogging
  with Common{

  import spark.implicits._

  override def transform(dataset: Dataset[_]): DataFrame = {
    logInfo(spark, "Transforming product metadata")
    val expandedCategories = expandCategories(dataset)
    val categoryMapped = applyCategoryMapping(expandedCategories)
    regroupCategories(categoryMapped).cache()
  }

  private def expandCategories(df: Dataset[_]): DataFrame = {
    logInfo(spark, "Expanding product metadata categories")
    df.select($"asin",
      explode($"categories").as("categories"),
      $"title",
      $"description")
      .select($"asin",
        explode($"categories").as("category"),
        trim(regexp_replace($"description", textBlacklistRegex, "")).as("description"),
        trim(regexp_replace($"title", textBlacklistRegex, "")).as("title"))
  }

  private def applyCategoryMapping(df: Dataset[_]): DataFrame = {
    logInfo(spark, "Applying category mapping to product metadata")
    val categoryMap = categoryParser.getCategoryMappings()
    val categories = categoryParser.getCategories()

    val firstCase = when($"category".isin(categoryMap(categories.head):_*), categories.head)
    val categoryColumn = categories.tail.foldLeft(firstCase){(column, category) =>
      column.when($"category".isin(categoryMap(category):_*), category)
    }

    df.select($"asin",
      trim($"title").as("title"),
      trim($"description").as("description"),
      categoryColumn.otherwise("skip").as("category")).
      where($"category".isNotNull && $"category" =!= "skip")
  }

  private def regroupCategories(df: Dataset[_]): DataFrame = {
    logInfo(spark, "Regrouping product categories")
    df.groupBy($"asin", $"title", $"description")
      .agg(collect_set($"category").as("categories"))
      .cache()
  }

  override def copy(extra: ParamMap): Transformer = ???

  override def transformSchema(schema: StructType): StructType = ???

  override val uid: String = Identifiable.randomUID("product-metadata-transformer")
}

@deprecated("Deprecated, use the transformer in the parent package")
object MetadataTransformer {
  def apply(spark: SparkSession,
            categoryParser: CategoryParser = CategoryParser(CategoryParser.defaultMappingPath)): MetadataTransformer =
    new MetadataTransformer(spark, categoryParser)
}
