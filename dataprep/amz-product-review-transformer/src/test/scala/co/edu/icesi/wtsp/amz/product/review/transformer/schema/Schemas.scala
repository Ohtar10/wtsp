package co.edu.icesi.wtsp.amz.product.review.transformer.schema

import org.apache.spark.sql.types.{StringType, StructField, StructType}

object Schemas {

  val documentSchema: StructType = new StructType()
    .add(StructField("category", StringType, nullable = false))
    .add(StructField("document", StringType, nullable = false))

}
