package co.edu.icesi.wtsp.util

import com.esri.core.geometry.ogc.OGCGeometry
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

object GeoUDF {

  /**
    * Takes the given GeoJSON string and converts it
    * to the WKT representation.
    *
    * If the input is null, then null will be returned
    *
    */
  val wktFromGeoJson: String => String = (json: String) => {
    if (json != null)
      OGCGeometry.fromGeoJson(json).asText()
    else
      null
  }

  val stWKTFromGeoJson: UserDefinedFunction = udf(wktFromGeoJson)
}

