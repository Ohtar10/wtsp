package co.edu.icesi.wtsp.tweets.transformer.schema

import co.edu.icesi.wtsp.util.GeoUDF
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions._


/**
  * Contains all the schemas needed for this job.
  *
  * Since once a dataframe is created the schema can't
  * be changed but an array of columns can be passed
  * to a select statement to get a new schema. The schemas
  * in this object are the Column arrays that creates
  * the new schemas instead of the spark struct types.
  *
  */
object Schemas{

  private val datePattern: String = "EEE MMM dd HH:mm:ss ZZZZZ yyyy"

  /**
    * The general tweet object schema.
    *
    * This is the common tweet schema used
    * for all the job phases.
    */
  val tweetObject: Seq[Column] = Seq(
    new Column("id"),
    new Column("text").as("tweet"),
    nanvl(new Column("favorite_count"), lit(0.0)).as("favorite_count"),
    nanvl(new Column("reply_count"), lit(0.0)).as("reply_count"),
    nanvl(new Column("retweet_count"), lit(0.0)).as("retweet_count"),
    new Column("user.id").alias("user_id"),
    new Column("user.screen_name").alias("user_name"),
    nanvl(new Column("user.followers_count"), lit(0.0)).as("user_followers_count"),
    nanvl(new Column("user.friends_count"), lit(0.0)).as("user_following_count"),
    new Column("user.location").as("user_location"),
    to_timestamp(new Column("created_at"), datePattern).as("created_timestamp"),
    year(new Column("created_timestamp")).as("year"),
    month(new Column("created_timestamp")).as("month"),
    dayofmonth(new Column("created_timestamp")).as("day"),
    hour(new Column("created_timestamp")).as("hour"),
    concat_ws(";", new Column("entities.hashtags.text")).as("hashtags"),
    concat_ws(";", new Column("entities.user_mentions.screen_name")).as("user_mentions"),
    concat_ws(";", new Column("entities.user_mentions.id")).as("user_id_mentions"),
    concat_ws(";", new Column("entities.urls.expanded_url")).as("expanded_urls"),
    GeoUDF.stWKTFromGeoJson(to_json(new Column("coordinates"))).as("location_geometry"),
    GeoUDF.stWKTFromGeoJson(to_json(new Column("place.bounding_box"))).as("place_geometry"),
    new Column("place.id").as("place_id"),
    new Column("place.name").as("place_name"),
    new Column("place.full_name").as("place_full_name"),
    new Column("place.country").as("country"),
    new Column("place.country_code").as("country_code"),
    new Column("place.place_type").as("place_type"),
    new Column("place.url").as("place_url")
  )

  /**
    * The columns required as input
    * for the tweet spam assassin pipeline.
    *
    * It assumes the input data frame complies
    * with the tweetObject general schema.
    */
  val tweetSpamObject: Seq[Column] = Seq(
    new Column("id").alias("Id"),
    new Column("text").alias("Tweet"),
    new Column("user_followers_count").alias("followers"),
    new Column("user_following_count").alias("following"),
    coalesce(new Column("retweet_count"), lit(0))
      .+(coalesce(new Column("favorite_count"), lit(0)))
      .+(coalesce(new Column("reply_count"), lit(0)))
      .alias("actions"),
    new Column("user_location").alias("location"),
    when(new Column("location").isNull, 0.0)
      .otherwise(1.0)
      .alias("has_location")
  )

}