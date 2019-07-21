package co.edu.icesi.wtsp.util

import com.esri.core.geometry.ogc.OGCGeometry
import org.apache.spark.internal.Logging
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

import scala.util.{Failure, Success, Try}

trait EsriOperable extends Serializable {
  def fromGeoJson(json: String): String = OGCGeometry.fromGeoJson(json).asText()
}

class GeoUDF extends EsriOperable with Logging with Serializable {

  /**
    * Takes the given GeoJSON string and converts it
    * to the WKT representation.
    *
    * If the input is null, then null will be returned
    *
    */
  val wktFromGeoJson: String => String = (json: String) => {
      Try(fromGeoJson(json)) match {
        case Success(value) => value
        case Failure(exception) => logDebug(s"Failed parsing GeoJson String $json", exception); null
      }
  }
}

object GeoUDF extends Serializable {
  val geoUDF = new GeoUDF()
  val stWKTFromGeoJson: UserDefinedFunction = udf(geoUDF.wktFromGeoJson)
}

