package co.edu.icesi.wtsp.util

import com.esri.core.geometry.ogc.OGCGeometry
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

object UDF {

  val wktFromGeoJson: String => String = (json: String) => {
    if (json != null)
      try{
        OGCGeometry.fromGeoJson(json).asText()
      }
      catch {
        case e: NullPointerException => e.printStackTrace(); ""
      }

    else
      ""
  }

  val stWKTFromGeoJson: UserDefinedFunction = udf(wktFromGeoJson)
}

