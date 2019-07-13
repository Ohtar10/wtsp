package co.edu.icesi.wtsp.tweets.transformer.job

/**
  * Job trait to identify
  * executable spark jobs.
  *
  */
trait Job {

  def execute(): Unit

}
