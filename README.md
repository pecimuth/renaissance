
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
$ java -jar 'renaissance-gpl-0.13.0.jar' <benchmarks>
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


### Licensing

The Renaissance Suite comes in two distributions,
and is available under both the MIT license and the GPL3 license.
The GPL distribution with all the benchmarks is licensed under the GPL3 license,
while the MIT distribution includes only those benchmarks that themselves
have less restrictive licenses.

Depending on your needs, you can use either of the two distributions.

The list below contains the licensing information (and JVM version requirements)
for each benchmark.


### List of benchmarks

The following is the complete list of benchmarks, separated into groups.

#### database

- `db-shootout` - Executes a shootout test using several in-memory databases.
  \
  Default repetitions: 16; APACHE2 license, MIT distribution; Supported JVM: 1.8 - 11



The suite also contains a group of benchmarks intended solely for testing
purposes:

#### dummy

- `dummy-empty` - A dummy benchmark which only serves to test the harness.
  \
  Default repetitions: 20; MIT license, MIT distribution; Supported JVM: 1.8 and later

- `dummy-failing` - A dummy benchmark for testing the harness (fails during iteration).
  \
  Default repetitions: 20; MIT license, MIT distribution; Supported JVM: 1.8 and later

- `dummy-param` - A dummy benchmark for testing the harness (test configurable parameters).
  \
  Default repetitions: 20; MIT license, MIT distribution; Supported JVM: 1.8 and later

- `dummy-setup-failing` - A dummy benchmark for testing the harness (fails during setup).
  \
  Default repetitions: 20; MIT license, MIT distribution; Supported JVM: 1.8 and later

- `dummy-teardown-failing` - A dummy benchmark for testing the harness (fails during teardown).
  \
  Default repetitions: 20; MIT license, MIT distribution; Supported JVM: 1.8 and later

- `dummy-validation-failing` - A dummy benchmark for testing the harness (fails during validation).
  \
  Default repetitions: 20; MIT license, MIT distribution; Supported JVM: 1.8 and later



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
See their respective READMEs under the `plugin` directory for more information.

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


### Complete list of command-line options

The following is a complete list of command-line options.

```
Renaissance Benchmark Suite, version 0.13.0
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


### JMH support

You can also build and run Renaissance with JMH. To build a JMH-enabled JAR, run:

```
$ tools/sbt/bin/sbt renaissanceJmh/jmh:assembly
```

To run the benchmarks using JMH, you can execute the following `java` command:

```
$ java -jar 'renaissance-jmh/target/scala-2.12/renaissance-jmh-assembly-0.13.0.jar'
```


### Contributing

Please see the [contribution guide](CONTRIBUTING.md) for a description of the contribution process.


### Documentation

Apart from documentation embedded directly in the source code, further
information about design and internals of the suite can be found in the
`documentation` folder of this project.


### Support

When you find a bug in the suite, when you want to propose a new benchmark or
ask for help, please, open an [Issue](https://github.com/renaissance-benchmarks/renaissance/issues)
at the project page at GitHub.

