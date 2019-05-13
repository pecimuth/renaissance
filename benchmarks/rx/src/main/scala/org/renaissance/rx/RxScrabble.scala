package org.renaissance.rx

import org.renaissance.Benchmark._
import org.renaissance.{Config, License, RenaissanceBenchmark}

class RxScrabble extends RenaissanceBenchmark {
  def description = "Solves the Scrabble puzzle using the Rx streams."

  override def defaultRepetitions = 80
  // TODO: Consolidate benchmark parameters across the suite.
  //  See: https://github.com/renaissance-benchmarks/renaissance/issues/27

  def licenses = License.create(License.GPL2)
  var shakespearePath: String = "/shakespeare.txt"

  var scrabblePath: String = "/scrabble.txt"

  var bench: RxScrabbleImplementation = null

  override def setUpBeforeAll(c: Config): Unit = {
    if (c.functionalTest) {
      shakespearePath = "/shakespeare-truncated.txt"
    }
    bench = new RxScrabbleImplementation(scrabblePath, shakespearePath)
  }

  override def runIteration(c: Config): Unit = {
    blackHole(bench.runScrabble().size())
  }
}
