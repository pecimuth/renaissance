lazy val scalaSAT = RootProject(uri("../."))

lazy val scalaCafeSAT = (project in file("."))
  .settings(
    name := "CafeSat",
    organization := "com.regblanc",
    version := "0.01",
    scalaVersion := (scalaSAT / scalaVersion).value,
    scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature"),
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.1.4" % "test",
    Test / parallelExecution := true,
  )
