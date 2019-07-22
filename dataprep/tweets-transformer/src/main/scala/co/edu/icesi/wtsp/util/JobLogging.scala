package co.edu.icesi.wtsp.util

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession

trait JobLogging extends Logging{

  def logInfo(spark: SparkSession, msg: String, context: Option[String] = None): Unit = context match {
      case Some(ctx) => {
        val message = s"[$ctx] $msg"
        logMessage(spark, message)
      }
      case None => logMessage(spark, msg)
      }


  private def logMessage(spark: SparkSession, msg: String): Unit = {
    logInfo(msg)
    spark.sparkContext.setJobDescription(msg)
  }

}
