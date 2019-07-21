// give the user a nice default project!

val circeVersion = "0.11.1"

lazy val app = (project in file(".")).

  settings(
    inThisBuild(List(
      organization := "co.edu.icesi.wtsp",
      scalaVersion := "2.11.8"
    )),
    name := "tweets-transformer",
    version := "0.2.0",

    sparkVersion := "2.4.3",
    sparkComponents := Seq(),

    javacOptions ++= Seq("-source", "1.8", "-target", "1.8"),
    javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled"),
    scalacOptions ++= Seq("-deprecation", "-unchecked"),
    parallelExecution in Test := false,
    fork := true,
    assemblyJarName in assembly := s"tweets-transformer-with-deps-${version.value}.jar",

    coverageHighlighting := true,

    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-streaming" % "2.4.3" % "provided",
      "org.apache.spark" %% "spark-sql" % "2.4.3" % "provided",
      "org.apache.spark" %% "spark-mllib" % "2.4.3" % "provided",
      "log4j" % "log4j" % "1.2.17" % "provided",

      "org.scalatest" %% "scalatest" % "3.0.1" % "test",
      "org.scalacheck" %% "scalacheck" % "1.13.4" % "test",
      "com.holdenkarau" %% "spark-testing-base" % "2.4.3_0.12.0" % "test",
      "org.mockito" %% "mockito-scala" % "0.4.5" % "test",

      "com.esri.geometry" % "esri-geometry-api" % "2.2.2",
      "com.github.scopt" %% "scopt" % "3.7.1"

    ),

    // uses compile classpath for the run task, including "provided" jar (cf http://stackoverflow.com/a/21803413/3827)
    run in Compile := Defaults.runTask(fullClasspath in Compile, mainClass in (Compile, run), runner in (Compile, run)).evaluated,

    scalacOptions ++= Seq("-deprecation", "-unchecked"),
    pomIncludeRepository := { x => false },

   resolvers ++= Seq(
      "sonatype-releases" at "https://oss.sonatype.org/content/repositories/releases/",
      "Typesafe repository" at "http://repo.typesafe.com/typesafe/releases/",
      "Second Typesafe repo" at "http://repo.typesafe.com/typesafe/maven-releases/",
      Resolver.sonatypeRepo("public")
    ),

    pomIncludeRepository := { x => false },

    // publish settings
    publishTo := {
      val nexus = "https://oss.sonatype.org/"
      if (isSnapshot.value)
        Some("snapshots" at nexus + "content/repositories/snapshots")
      else
        Some("releases"  at nexus + "service/local/staging/deploy/maven2")
    }
  )
