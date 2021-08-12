
## Renaissance Benchmark Suite

<p align="center">
  <img height="180px" src="https://github.com/renaissance-benchmarks/renaissance/raw/master/website/resources/images/mona-lisa-round.png"/>
</p>


The Renaissance Benchmark Suite aggregates common modern JVM workloads,
including, but not limited to, Big Data, machine-learning, and functional programming.
The suite is intended to be used to optimize just-in-time compilers, interpreters, GCs,
and for tools such as profilers, debuggers, or static analyzers, and even different hardware.
It is intended to be an open-source, collaborative project,
in which the community can propose and improve benchmark workloads.


### Running the suite

To run the suite, you will need to download a Renaissance Suite JAR
from <https://renaissance.dev/download>.
If you wish to build it yourself, please, consult [CONTRIBUTING.md](CONTRIBUTING.md)
for instructions on building.

To run a Renaissance benchmark, you need to have a JRE installed.
This allows you to execute the following `java` command:


```
$ java -jar 'renaissance-gpl-0.12.0.jar' <benchmarks>
```

Above, `<benchmarks>` is the list of benchmarks that you wish to run.
For example, you can specify `scala-kmeans` as the benchmark.

The suite generally executes the benchmark's measured operation multiple times. By default,
the suite executes each benchmark operation for a specific number of times. The benchmark-specific
number of repetitions is only intended for quick visual evaluation of benchmark execution time,
but is not sufficient for thorough experimental evaluation, which will generally need much more
repetitions.

For thorough experimental evaluation, the benchmarks should be repeated for a large number of
times or executed for a long time. The number of repetitions and the execution time can be
set for all benchmarks using the `-r` or `-t` options. More fine-grained control over benchmark
execution can be achieved by providing the harness with a plugin implementing a custom execution
policy (see [below](#plugins) for details).


#### Complete list of command-line options

The following is a complete list of command-line options.

```
Renaissance Benchmark Suite, version 0.12.0
Usage: renaissance [options] [benchmark-specification]

  -h, --help               Prints this usage text.
  -r, --repetitions <count>
                           Execute the measured operation a fixed number of times.
  -t, --run-seconds <seconds>
                           Execute the measured operation for fixed time (wall-clock).
  --operation-run-seconds <seconds>
                           Execute the measured operation for fixed accumulated operation time (wall-clock).
  --policy <class-path>!<class-name>
                           Use policy plugin to control repetition of measured operation execution.
  --plugin <class-path>[!<class-name>]
                           Load external plugin. Can appear multiple times.
  --with-arg <value>       Adds an argument to the plugin or policy specified last. Can appear multiple times.
  --csv <csv-file>         Output results as CSV to <csv-file>.
  --json <json-file>       Output results as JSON to <json-file>.
  -c, --configuration <conf-name>
                           Use benchmark parameters from configuration <conf-name>.
  -o, --override <name>=<value>
                           Override the value of a configuration parameter <name> to <value>.
  --scratch-base <dir>     Create scratch directories in <dir>. Defaults to current directory.
  --keep-scratch           Keep the scratch directories after VM exit. Defaults to deleting scratch directories.
  --no-forced-gc           Do not force garbage collection before each measured operation. Defaults to forced GC.
  --no-jvm-check           Do not check benchmark JVM version requirements (for execution or raw-list).
  --list                   Print the names and descriptions of all benchmarks.
  --raw-list               Print the names of benchmarks compatible with this JVM (one per line).
  --group-list             Print the names of all benchmark groups (one per line).
  benchmark-specification  List of benchmarks (or groups) to execute (or 'all').

```


### List of benchmarks

The following is the complete list of benchmarks, separated into groups.

#### apache-spark

- `als` - Runs the ALS algorithm from the Spark ML library. (default repetitions: 30)

- `chi-square` - Runs the chi-square test from Spark MLlib. (default repetitions: 60)

- `dec-tree` - Runs the Random Forest algorithm from the Spark ML library. (default repetitions: 40)

- `gauss-mix` - Computes a Gaussian mixture model using expectation-maximization. (default repetitions: 40)

- `log-regression` - Runs the Logistic Regression algorithm from the Spark ML library. (default repetitions: 20)

- `movie-lens` - Recommends movies using the ALS algorithm. (default repetitions: 20)

- `naive-bayes` - Runs the multinomial Naive Bayes algorithm from the Spark ML library. (default repetitions: 30)

- `page-rank` - Runs a number of PageRank iterations, using RDDs. (default repetitions: 20)

#### concurrency

- `akka-uct` - Runs the Unbalanced Cobwebbed Tree actor workload in Akka. (default repetitions: 24)

- `fj-kmeans` - Runs the k-means algorithm using the fork/join framework. (default repetitions: 30)

- `reactors` - Runs benchmarks inspired by the Savina microbenchmark workloads in a sequence on Reactors.IO. (default repetitions: 10)

#### database

- `db-shootout` - Executes a shootout test using several in-memory databases. (default repetitions: 16)

- `neo4j-analytics` - Executes Neo4J graph queries against a movie database. (default repetitions: 20)

#### functional

- `future-genetic` - Runs a genetic algorithm using the Jenetics library and futures. (default repetitions: 50)

- `mnemonics` - Solves the phone mnemonics problem using JDK streams. (default repetitions: 16)

- `par-mnemonics` - Solves the phone mnemonics problem using parallel JDK streams. (default repetitions: 16)

- `rx-scrabble` - Solves the Scrabble puzzle using the Rx streams. (default repetitions: 80)

- `scrabble` - Solves the Scrabble puzzle using JDK Streams. (default repetitions: 50)

#### scala

- `dotty` - Runs the Dotty compiler on a set of source code files. (default repetitions: 50)

- `philosophers` - Solves a variant of the dining philosophers problem using ScalaSTM. (default repetitions: 30)

- `scala-doku` - Solves Sudoku Puzzles using Scala collections. (default repetitions: 20)

- `scala-kmeans` - Runs the K-Means algorithm using Scala collections. (default repetitions: 50)

- `scala-stm-bench7` - Runs the stmbench7 benchmark using ScalaSTM. (default repetitions: 60)

#### web

- `finagle-chirper` - Simulates a microblogging service using Twitter Finagle. (default repetitions: 90)

- `finagle-http` - Sends many small Finagle HTTP requests to a Finagle HTTP server and awaits response. (default repetitions: 12)



The suite also contains a group of benchmarks intended solely for testing
purposes:

#### dummy

- `dummy-empty` - A dummy benchmark which only serves to test the harness. (default repetitions: 20)

- `dummy-failing` - A dummy benchmark for testing the harness (fails during iteration). (default repetitions: 20)

- `dummy-param` - A dummy benchmark for testing the harness (test configurable parameters). (default repetitions: 20)

- `dummy-setup-failing` - A dummy benchmark for testing the harness (fails during setup). (default repetitions: 20)

- `dummy-teardown-failing` - A dummy benchmark for testing the harness (fails during teardown). (default repetitions: 20)

- `dummy-validation-failing` - A dummy benchmark for testing the harness (fails during validation). (default repetitions: 20)



### <a name="plugins">Using plugins to customize the harness</a>

If you are using an external tool to inspect a benchmark, such as an instrumentation agent,
or a profiler, then you may need to make this tool aware of when a benchmark's measured
operation is about to be executed and when it finished executing.

If you need to collect additional metrics associated with the execution of the measured
operation, e.g., hardware counters, you will need to be notified about operation execution,
and you may want to store the measured values in the output files produced by the harness.

If you need the harness to produce output files in different format (other than CSV or JSON),
you will need to be notified about values of metrics collected by the harness and other plugins.

If you need more fine-grained control over the repetition of the benchmark's measured operation,
you will need to be able to tell the harness when to keep executing the benchmark and when to
stop.

To this end, the suite provides hooks for plugins which can subscribe to events related to
harness state and benchmark execution.

This repository contains two such plugins: one that uses a native agent built with
[PAPI](https://icl.utk.edu/papi/) to collect information from hardware counters and
a plugin for collecting information from a
[CompilationMXBean](https://docs.oracle.com/javase/8/docs/api/java/lang/management/CompilationMXBean.html).

If you wish to create your own plugin, please consult
[documentation/plugins.md](documentation/plugins.md) for more details.

To make the harness use an external plugin, it needs to be specified on the command line.
The harness can load multiple plugins, and each must be enabled using the
`--plugin <class-path>[!<class-name>]` option. The `<class-path>` is the class path on which
to look for the plugin class (optionally, you may add `<class-name>` to specify a fully
qualified name of the plugin class).

Custom execution policy must be enabled using the `--policy <class-path>!<class-name>` option.
The syntax is the same as in case of normal plugins (and the policy is also a plugin, which
can register for all event types), but this option tells the harness to actually use the
plugin to control benchmark execution. Other than that, policy is treated the same way as a
plugin.

When registering plugins for pair events (harness init/shutdown, benchmark set up/tear down,
operation set up/tear down), the plugins specified earlier will "wrap" plugins specified later.
This means that for example plugins that want to collect additional measurements and need to
invoked as close as possible to the measured operation need to be specified last. Note that
this also applies to an external execution policy, which would be generally specified first,
but any order is possible.

Plugins (and policies) can receive additional command line arguments. Each argument must be
given using the `--with-arg <arg>` option, which appends `<arg>` to the list of arguments for
the plugin (or policy) that was last mentioned on the command line. Whenever a `--plugin`
(or `--policy`) option is encountered, the subsequent `--with-arg` options will append
arguments to that plugin (or policy).


### JMH support

You can also build and run Renaissance with JMH. To build a JMH-enabled JAR, run:

```
$ tools/sbt/bin/sbt renaissanceJmh/jmh:assembly
```

To run the benchmarks using JMH, you can execute the following `java` command:

```
$ java -jar 'renaissance-jmh/target/scala-2.12/renaissance-jmh-assembly-0.12.0.jar'
```


### Contributing

Please see the [contribution guide](CONTRIBUTING.md) for a description of the contribution process.


### Licensing

The Renaissance Suite comes in two distributions,
and is available under both the MIT license and the GPL3 license.
The GPL distribution with all the benchmarks is licensed under the GPL3 license,
while the MIT distribution includes only those benchmarks that themselves
have less restrictive licenses.

Depending on your needs, you can use either of the two distributions.
The following table contains the licensing information (and JVM version
requirements) for all the benchmarks:

| Benchmark        | Licenses   | Distro | JVM required (min) | JVM supported (max) |
| :--------------- | :--------- | :----: | :----------------: | :-----------------: |
| akka-uct | MIT | MIT | 1.8 |  |
| als | APACHE2 | MIT | 1.8 |  |
| chi-square | APACHE2 | MIT | 1.8 |  |
| db-shootout | APACHE2 | MIT | 1.8 | 11 |
| dec-tree | APACHE2 | MIT | 1.8 |  |
| dotty | BSD3 | MIT | 1.8 |  |
| finagle-chirper | APACHE2 | MIT | 1.8 |  |
| finagle-http | APACHE2 | MIT | 1.8 |  |
| fj-kmeans | APACHE2 | MIT | 1.8 |  |
| future-genetic | APACHE2 | MIT | 1.8 |  |
| gauss-mix | APACHE2 | MIT | 1.8 |  |
| log-regression | APACHE2 | MIT | 1.8 |  |
| mnemonics | MIT | MIT | 1.8 |  |
| movie-lens | APACHE2 | MIT | 1.8 |  |
| naive-bayes | APACHE2 | MIT | 1.8 |  |
| neo4j-analytics | GPL3 | GPL3 | 11 | 15 |
| page-rank | APACHE2 | MIT | 1.8 |  |
| par-mnemonics | MIT | MIT | 1.8 |  |
| philosophers | BSD3 | MIT | 1.8 |  |
| reactors | MIT | MIT | 1.8 |  |
| rx-scrabble | GPL2 | GPL3 | 1.8 |  |
| scala-doku | MIT | MIT | 1.8 |  |
| scala-kmeans | MIT | MIT | 1.8 |  |
| scala-stm-bench7 | BSD3, GPL2 | GPL3 | 1.8 |  |
| scrabble | GPL2 | GPL3 | 1.8 |  |


### Documentation

Apart from documentation embedded directly in the source code, further
information about design and internals of the suite can be found in the
`documentation` folder of this project.

