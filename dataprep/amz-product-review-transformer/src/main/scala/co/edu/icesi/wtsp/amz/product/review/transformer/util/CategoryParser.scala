package co.edu.icesi.wtsp.amz.product.review.transformer.util

import cats.syntax.either._
import io.circe._
import io.circe.generic.auto._
import io.circe.yaml

import scala.io.Source

case class Category(name: String, mappings: List[String])
case class CategoryMapping(categories: List[Category])

/**
 * Product Category parser
 */
class CategoryParser(categoryMapping: CategoryMapping) {

  /**
   * Parses the YAML file provided accoirding to
   * category specifications
   * @param path to the yaml file with the categories
   * @return
   */
  def parse(path: String): CategoryMapping = {

    val source = Source.fromFile(path)
    val content = try source.mkString finally source.close()

    val json = yaml.parser.parse(content)

    json.leftMap(err => err:Error).
      flatMap(_.as[CategoryMapping]).
      valueOr(throw _)
  }

  def getCategories(): List[String] = {
    categoryMapping.categories.map(category => category.name)
  }

  def getCategoryMappings(): Map[String, List[String]] = {
    categoryMapping.categories.map(category => category.name -> category.mappings).toMap
  }
}

object CategoryParser{
  val defaultMappingPath: String = "src/main/resources/mapping/category_mappings.yml"
  @deprecated("Deprecated, use fromYamlFile instead.")
  def apply(path: String): CategoryParser = {
    fromYamlFile(path)
  }

  def fromYamlString(yaml: String): CategoryParser = {
    new CategoryParser(parse(yaml))
  }

  def fromYamlFile(path: String): CategoryParser = {
    val source = Source.fromFile(path)
    val content = try source.mkString finally source.close()
    new CategoryParser(parse(content))
  }

  /**
   * Parses the YAML file provided accoirding to
   * category specifications
   * @param content to the yaml file with the categories
   * @return
   */
  private def parse(content: String): CategoryMapping = {
    val json = yaml.parser.parse(content)

    json.leftMap(err => err:Error).
      flatMap(_.as[CategoryMapping]).
      valueOr(throw _)
  }

}
