package co.edu.icesi.wtsp.amz.product.review.transformer.util

import java.io.FileNotFoundException

import co.edu.icesi.wtsp.amz.product.review.transformer.SpecCommon
import io.circe.DecodingFailure
import org.scalatest.{FlatSpec, Matchers}

class CategoryParserSpec extends FlatSpec
  with Matchers
  with SpecCommon{

  "The config parser" should "be able to parse valid configuration" in {
    val categoryParser = CategoryParser(categoryConfigPath)
    val categoryMapping = categoryParser.getCategoryMappings()

    //Check the corresponding keys are present
    val keys = categoryMapping.keySet
    keys should contain("Music")
    keys should contain("Movies & TV")
    keys should contain("Clothing, Shoes & Jewelry")

    //Check some of the keys for the values
    categoryMapping("Movies & TV").shouldBe(List("Movies", "Movies & TV"))
    categoryMapping("Music").shouldBe(List(
      "Music",
      "Digital Music",
      "CDs & Vinyl",
      "Pop",
      "Jazz",
      "Alternative Rock",
      "Rock",
      "World Music",
      "Dance & Electronic"
    ))

  }
  it should "be able to get the category list" in {
    val categoryParser = CategoryParser(categoryConfigPath)
    val categoryList = categoryParser.getCategories()

    //Check the corresponding keys are present
    categoryList should contain("Music")
    categoryList should contain("Movies & TV")
    categoryList should contain("Clothing, Shoes & Jewelry")
  }
  it should "fail when passed invalid configurations" in {
    val invalidMapping = s"$resourcesBasePath/config/invalid_category_mapping.yml"

    a [DecodingFailure] should be thrownBy {
      CategoryParser(invalidMapping)
    }
  }
  it should "fail when passed an invalid path" in {

    a [FileNotFoundException] should be thrownBy {
      val categoryParser = CategoryParser("invalid_path")
    }
  }
}
