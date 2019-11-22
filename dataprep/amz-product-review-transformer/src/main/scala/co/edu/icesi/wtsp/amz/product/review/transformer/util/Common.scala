package co.edu.icesi.wtsp.amz.product.review.transformer.util

/**
 * Common
 * Contains general variables and utilities for the pipeline
 */
trait Common {
  /**
   * Regex pattern to clean up text from dirty character.
   */
  val textBlacklistRegex: String = "&.+;|'+|\"+|[@#$%^&*<>_-]"

  /**
   * The minimum amount of characters for the document text.
   */
  val documentTextMinCharacters: Int = 100
}
