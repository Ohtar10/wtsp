package co.edu.icesi.wtsp.amz.product.review.transformer.job

case class StepArgs(map: Map[String, _])

trait Step {
  def validate(input: StepArgs): Unit
  def process(input: StepArgs): StepArgs
  def save(input: StepArgs):Unit
}
