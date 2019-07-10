package co.edu.icesi.wtsp.tweets.transformer

import java.io.File

/**
  * Contain common variables and utility methods
  * shared accross all the test suites.
  *
  */
trait SpecCommon {

  /**
    * Base path for the models
    */
  val modelBasePath = "src/test/resources/models/"

  /**
    * Path to the TF based spam assassin model
    */
  val tfPipeline = s"${modelBasePath}spark/tweet-spam-assassin-tf"

  /**
    * Path to the Word 2 Vec based spam assassin model
    */
  val w2vPipeline = s"${modelBasePath}spark/tweet-spam-assassin-w2v"

  /**
    * Path to the spam assassin tweet test data set.
    */
  val tweetsPath = "src/test/resources/models/tweets/test.csv"

  /**
    * Contain tweet files as JSON format
    * just as if they were downloaded from
    * the Twitter API
    */
  val rawTweetsPath = "src/test/resources/tweets/*/"

  /**
    * The output path for the spark related
    * operations.
    */
  val testOutputPath = "src/test/resources/output/"

  /**
    * Deletes recursively the files specified
    * at the given File object.
    *
    * @param file the path to delete recursively
    */
  def deleteRecursively(file: File): Unit = {
    if (file.isDirectory)
    {
      file.listFiles().foreach(deleteRecursively)
    }
    if (file.exists && !file.delete)
    {
      throw new Exception(s"Unable to delete file: ${file.getAbsolutePath}")
    }
  }

}
