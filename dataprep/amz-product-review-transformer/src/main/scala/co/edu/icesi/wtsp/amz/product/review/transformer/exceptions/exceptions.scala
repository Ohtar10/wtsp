package co.edu.icesi.wtsp.amz.product.review.transformer.exceptions

class InvalidCategoryMappingException(private val message: String,
                                                 private val cause: Throwable)
  extends RuntimeException(message, cause)

class StepExecutionException(private val message: String,
                                        private val cause: Throwable)
  extends RuntimeException(message, cause)

class InvalidStepArgumentException(private val message: String,
                                        private val cause: Throwable = null)
  extends StepExecutionException(message, cause)