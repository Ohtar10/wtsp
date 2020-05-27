package co.edu.icesi.wtsp.amz.product.review.transformer.util

import co.edu.icesi.wtsp.amz.product.review.transformer.SpecCommon
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.{FlatSpec, Matchers}

class TransformCategoriesUDFSpec extends FlatSpec
  with Matchers
  with DataFrameSuiteBase
  with SpecCommon {

  import spark.implicits._

  case class DummyRecord(categories: Array[String], document: String)
  case class DummyResult(categories: String, document: String)

  val categoryParser: CategoryParser = CategoryParser.fromYamlString(categoryMappingsYaml)

  "Transform Categories UDF" should "be able to process multi category columns" in {
    val testDF = spark.createDataFrame(List(
      DummyRecord(Array("Movies", "Movies & TV", "Clothing"), "This is an awesome movie"),
      DummyRecord(Array("Shirts", "Boots", "Movies"), "These boots are made for walking!"),
      DummyRecord(Array("lalala"), "This should result in empty category")
    ))

    val expected = spark.createDataFrame(List(
      DummyResult("Clothing, Shoes & Jewelry;Movies & TV", "This is an awesome movie"),
      DummyResult("Clothing, Shoes & Jewelry;Movies & TV", "These boots are made for walking!"),
      DummyResult("", "This should result in empty category")
    ))

    val transform_categories = TransformCategoryUDF.build(categoryParser)
    val transformed = testDF.select(transform_categories($"categories").as("categories"), $"document")

    assertDataFrameEquals(transformed.sort($"categories"), expected.sort($"categories"))
  }
  it should "be able to process single category columns" in {
    val testDF = spark.createDataFrame(List(
      DummyRecord(Array("Movies"), "This is an awesome movie"),
      DummyRecord(Array("Boots"), "These boots are made for walking!"),
      DummyRecord(Array("lalala"), "This should result in empty category")
    ))

    val expected = spark.createDataFrame(List(
      DummyResult("Movies & TV", "This is an awesome movie"),
      DummyResult("Clothing, Shoes & Jewelry", "These boots are made for walking!"),
      DummyResult("", "This should result in empty category")
    ))

    val transform_categories = TransformCategoryUDF.build(categoryParser)
    val transformed = testDF.select(transform_categories($"categories").as("categories"), $"document")

    assertDataFrameEquals(transformed.sort($"categories"), expected.sort($"categories"))
  }
  it should "be able to process mix of multi and single categories at once" in {
    val testDF = spark.createDataFrame(List(
      DummyRecord(Array("Movies", "Movies & TV", "Clothing"), "This is an awesome movie"),
      DummyRecord(Array("Boots"), "These boots are made for walking!"),
      DummyRecord(Array("lalala"), "This should result in empty category")
    ))

    val expected = spark.createDataFrame(List(
      DummyResult("Clothing, Shoes & Jewelry;Movies & TV", "This is an awesome movie"),
      DummyResult("Clothing, Shoes & Jewelry", "These boots are made for walking!"),
      DummyResult("", "This should result in empty category")
    ))

    val transform_categories = TransformCategoryUDF.build(categoryParser)
    val transformed = testDF.select(transform_categories($"categories").as("categories"), $"document")

    assertDataFrameEquals(transformed.sort($"categories"), expected.sort($"categories"))
  }

}
