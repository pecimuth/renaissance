package org.renaissance.neo4j

import java.nio.file.Paths

import org.renaissance.Benchmark._
import org.renaissance.neo4j.analytics.AnalyticsBenchmark
import org.renaissance.{Config, License, RenaissanceBenchmark}

class Neo4jAnalytics extends RenaissanceBenchmark {
  def description = "Executes Neo4J graph queries against a movie database."

  def licenses = License.create(License.GPL3)
  // TODO: Unify handling of scratch directories throughout the suite.
  //  See: https://github.com/renaissance-benchmarks/renaissance/issues/13

  val scratchPath = Paths.get("target", "modules", "neo4j", "neo4j-analytics.db")

  // TODO: Consolidate benchmark parameters across the suite.
  //  See: https://github.com/renaissance-benchmarks/renaissance/issues/27

  var benchmark: AnalyticsBenchmark = new AnalyticsBenchmark(
    scratchPath.toFile,
    sys.props.get("renaissance.neo4j.long-query-count").map(_.toInt),
    sys.props.get("renaissance.neo4j.short-query-count").map(_.toInt),
    sys.props.get("renaissance.neo4j.mutator-query-count").map(_.toInt)
  )

  override def defaultRepetitions = 20
  override def setUpBeforeAll(c: Config): Unit = {
    benchmark.setupAll()
  }

  override def tearDownAfterAll(c: Config): Unit = {
    benchmark.tearAll()
  }

  protected def runIteration(config: Config): Unit = {
    benchmark.run()
  }
}
