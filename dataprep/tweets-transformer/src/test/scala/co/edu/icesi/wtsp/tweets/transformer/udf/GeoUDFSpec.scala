package co.edu.icesi.wtsp.tweets.transformer.udf

import org.scalatest.{FlatSpec, Matchers}
import co.edu.icesi.wtsp.util.GeoUDF

class GeoUDFSpec extends FlatSpec
    with Matchers{

  "The wktFromGeoJson function" should "convert a point in geoJson into wkt" in {
    val geoJsonExamples = Map(
      ("POINT (-76.52080535888672 3.440770613885366)",
      "{\"type\": \"Point\", " +
        "\"coordinates\": [-76.52080535888672, 3.440770613885366]" +
        "}"),
      ("POLYGON ((-76.53067588806152 3.4392498642430245, " +
        "-76.52837991714478 3.4392498642430245, " +
        "-76.52837991714478 3.441670211122996, " +
        "-76.53067588806152 3.441670211122996, " +
        "-76.53067588806152 3.4392498642430245))",
        "{\"type\": \"Polygon\"," +
          "\"coordinates\": [[[-76.53067588806152, 3.4392498642430245], " +
          "[-76.52837991714478, 3.4392498642430245]," +
          "[-76.52837991714478, 3.441670211122996]," +
          "[-76.53067588806152, 3.441670211122996]," +
          "[-76.53067588806152, 3.4392498642430245]]]}")
    )

    geoJsonExamples.foreach{ geometry =>
      GeoUDF.wktFromGeoJson(geometry._2) shouldBe geometry._1
    }
  }
  it should "return null when a null geojson is given" in {
    GeoUDF.wktFromGeoJson(null) shouldBe null
  }
}
