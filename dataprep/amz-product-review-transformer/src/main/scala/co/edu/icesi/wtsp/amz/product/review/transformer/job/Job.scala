package co.edu.icesi.wtsp.amz.product.review.transformer.job

/**
 * Job trait to identify
 * executable spark jobs.
 *
 */
trait Job {

  def execute(): Unit

}
