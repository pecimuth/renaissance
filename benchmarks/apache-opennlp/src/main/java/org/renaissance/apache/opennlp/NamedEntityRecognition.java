package org.renaissance.apache.opennlp;

import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;
import org.renaissance.Benchmark;
import org.renaissance.BenchmarkContext;
import org.renaissance.BenchmarkResult;
import org.renaissance.BenchmarkResult.Validators;
import org.renaissance.License;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.stream.Collectors;

import static java.util.Objects.requireNonNull;
import static org.renaissance.Benchmark.*;

@Name("named-entity-recognition")
@Group("opennlp")
@Summary("A benchmark that tokenizes an input text and performs " +
         "named entity recognition using maximum-entropy-based models.")
@Licenses(License.APACHE2)
@Repetitions(10)
@Configuration(name = "test")
public final class NamedEntityRecognition implements Benchmark {
  private TokenNameFinderModel nameFinderModel;
  private TokenizerModel tokenizerModel;
  private String input;

  @Override
  public void setUpBeforeAll(BenchmarkContext context) {
    input = readFile("/shakespeare-truncated.txt");
    InputStream inputStreamTokenizer = getInputStream("/en-token.bin");
    InputStream inputStreamNer = getInputStream("/en-ner-person.bin");
    try {
      tokenizerModel = new TokenizerModel(inputStreamTokenizer);
      nameFinderModel = new TokenNameFinderModel(inputStreamNer);
    } catch (IOException e) {
      throw new RuntimeException(e.getMessage());
    }
  }

  private InputStream getInputStream(String resourceName) {
    return requireNonNull(getClass().getResourceAsStream(resourceName));
  }

  private String readFile(String resourceName) {
    return new BufferedReader(new InputStreamReader(getInputStream(resourceName)))
            .lines()
            .collect(Collectors.joining("\n"));
  }

  @Override
  public BenchmarkResult run(BenchmarkContext c) {
    TokenizerME tokenizer = new TokenizerME(tokenizerModel);
    String[] tokens = tokenizer.tokenize(input);
    NameFinderME nameFinder = new NameFinderME(nameFinderModel);
    Span[] spans = nameFinder.find(tokens);
    return Validators.simple(
            "The number of recognized named entities",
            160,
            spans.length
    );
  }
}
