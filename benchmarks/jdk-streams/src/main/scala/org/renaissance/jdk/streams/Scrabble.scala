package org.renaissance.jdk.streams

import org.renaissance.Benchmark._
import org.renaissance.{Config, License, RenaissanceBenchmark}

class Scrabble extends RenaissanceBenchmark {
  def description = "Solves the Scrabble puzzle using JDK Streams."

  override def defaultRepetitions = 50

  def licenses = License.create(License.GPL2)
  // TODO: Consolidate benchmark parameters across the suite.
  //  See: https://github.com/renaissance-benchmarks/renaissance/issues/27

  var shakespearePath = "/shakespeare.txt"

  var scrabblePath = "/scrabble.txt"

  var scrabble: JavaScrabble = null

  override def setUpBeforeAll(c: Config): Unit = {
    if (c.functionalTest) {
      shakespearePath = "/shakespeare-truncated.txt"
    }
    scrabble = new JavaScrabble(shakespearePath, scrabblePath)
  }

  override def runIteration(c: Config): Unit = {
    blackHole(scrabble.run())
  }
}
