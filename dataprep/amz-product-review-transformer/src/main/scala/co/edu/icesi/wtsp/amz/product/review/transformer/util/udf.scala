package co.edu.icesi.wtsp.amz.product.review.transformer.util

import org.apache.spark.internal.Logging
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

class TransformCategoryUDF(categoryParser: CategoryParser, separator: String) extends Serializable with Logging {

  val categoryMap: Map[String, List[String]] = categoryParser.getCategoryMappings()
  val validCategories: List[String] = categoryMap.values.flatten.toList
  val inverseMap: Map[String, String] = categoryMap.map(item => item._2 -> item._1)
    .flatMap(item => item._1.map(vc => vc -> item._2))

  val transformCategoryFn: Seq[String] => String = (categoryList: Seq[String]) => {
    if (categoryList != null)
      categoryList.filter(v => validCategories.contains(v)).map(v => inverseMap(v)).sorted.toSet.mkString(separator)
    else
      ""
  }

}

object TransformCategoryUDF extends Serializable {
  def build(categoryParser: CategoryParser, separator: String = ";"): UserDefinedFunction = {
    udf(new TransformCategoryUDF(categoryParser, separator).transformCategoryFn)
  }
}