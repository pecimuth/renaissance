package org.renaissance;

import java.util.Arrays;

/**
 * Represents the result of one benchmark execution. Each benchmark should
 * return a result the validity of which can be checked. The harness does not
 * care too much about the content, only that it can ask the benchmark to
 * validate its own result. A benchmark should store all necessary information
 * in the object implementing this interface.
 */
public interface BenchmarkResult {

  /**
   * Validates this benchmark result. Benchmarks are not expected to fail, but
   * if the result is invalid, throws this method must throw an exception
   * exception with an error message providing details on why the validation
   * failed.
   *
   * @throws ValidationException if the result is invalid.
   */
  void validate() throws ValidationException;

  //

  @Deprecated
  static BenchmarkResult dummy(Object... objects) {
    return () -> {
      Arrays.fill(objects, null);

      System.err.println("WARNING: This benchmark provides no result that can be validated.");
      System.err.println("         There is no way to check that no silent failure occurred.");
    };
  }


  static BenchmarkResult compound(final BenchmarkResult... results) {
    assert results != null;

    return () -> {
      for (final BenchmarkResult result : results) {
        result.validate();
      }
    };
  }


  static BenchmarkResult simple(
    final String name, final long expected, final long actual
  ) {
      return () -> ValidationException.throwIfNotEqual(expected, actual, name);
  }


  static BenchmarkResult simple(
    final String name, final double expected, final double actual, final double epsilon
  ) {
    return () -> ValidationException.throwIfNotEqual(expected, actual, epsilon, name);
  }

}
