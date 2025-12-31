package app;

import java.util.Map;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
@RestController
public class Application {

  public static void main(String[] args) {
    SpringApplication.run(Application.class, args);
  }

  @GetMapping("/thread")
  public String thread() {
    return Thread.currentThread().toString();
  }

  @GetMapping("/")
  public Map<String, String> hello() {
    return Map.of("hello", "world");
  }
}
