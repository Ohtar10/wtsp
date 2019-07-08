package co.edu.icesi.wtsp.tweets.transformer

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

}
