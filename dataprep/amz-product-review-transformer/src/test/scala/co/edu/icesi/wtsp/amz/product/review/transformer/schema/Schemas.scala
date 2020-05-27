package co.edu.icesi.wtsp.amz.product.review.transformer.schema

import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}

object Schemas {

  val documentSchema: StructType = new StructType()
    .add(StructField("categories", StringType, nullable = true))
    .add(StructField("document", StringType, nullable = false))

  val multiCategorySchema: StructType = new StructType()
    .add(StructField("categories", ArrayType(StringType), nullable = false))
    .add(StructField("document", StringType, nullable = false))
}
