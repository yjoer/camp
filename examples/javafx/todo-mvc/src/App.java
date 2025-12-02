import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class App extends Application {

  @Override
  public void start(Stage stage) {
    TodoController controller = new TodoController();
    Scene scene = new Scene(controller.view(), 640, 480);

    stage.setTitle("Todo MVC");
    stage.setScene(scene);
    stage.show();
  }
}
