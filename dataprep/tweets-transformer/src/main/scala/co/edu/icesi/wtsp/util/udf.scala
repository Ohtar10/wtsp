package co.edu.icesi.wtsp.util

import com.esri.core.geometry.ogc.OGCGeometry
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

object UDF {

  val wktFromGeoJson: String => String = (json: String) => {
    if (json != null)
      OGCGeometry.fromGeoJson(json).asText()
    else
      ""
  }

  val stWKTFromGeoJson: UserDefinedFunction = udf(wktFromGeoJson)
}

