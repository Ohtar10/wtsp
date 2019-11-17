package co.edu.icesi.wtsp.amz.product.review.transformer.util

import cats.syntax.either._
import io.circe._
import io.circe.generic.auto._
import io.circe.yaml

import scala.io.Source

/**
 * Product Category parser
 */
class CategoryParser {

  case class Category(name: String, mappings: List[String])
  case class CategoryMapping(categories: List[Category])

  private var categoryMapping: CategoryMapping = CategoryMapping(List())

  def this(path: String){
    this()
    this.categoryMapping = parse(path)
  }

  /**
   * Parses the YAML file provided accoirding to
   * category specifications
   * @param path to the yaml file with the categories
   * @return
   */
  def parse(path: String): CategoryMapping ={

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
  def apply(path: String): CategoryParser = new CategoryParser(path)
}
