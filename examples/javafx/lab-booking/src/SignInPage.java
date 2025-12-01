import java.io.IOException;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.PasswordField;
import javafx.scene.control.TextField;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import lib.AppState;
import lib.DataRepository;

class SignInPage extends Scene {

  SignInPage() {
    super(on_create(), 800, 600);
  }

  private static Parent on_create() {
    try {
      FXMLLoader loader = new FXMLLoader(SignInPage.class.getResource("/sign-in.fxml"));
      loader.setController(new SignInController());

      return loader.load();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}

class SignInController {

  @FXML
  Text message;

  @FXML
  TextField email_address;

  @FXML
  PasswordField password;

  @FXML
  Button sign_up_button;

  @FXML
  Button sign_in_button;

  @FXML
  void handle_sign_up() {
    Stage stage = (Stage) sign_up_button.getScene().getWindow();
    stage.setScene(new SignUpPage());
  }

  @FXML
  void handle_sign_in() {
    if (email_address.getText().isEmpty()) {
      message.setText("Email address is required.");
      return;
    }

    if (password.getText().isEmpty()) {
      message.setText("Password is required.");
      return;
    }

    DataRepository repo = new DataRepository();
    String pw = repo.find_password_by_email(email_address.getText());

    if (pw == null || !pw.equals(password.getText())) {
      message.setText("Invalid email address or password.");
      return;
    }

    AppState state = AppState.get_instance();
    int user_id = repo.find_user_id_by_email(email_address.getText());
    state.set_user_id(user_id);

    Stage stage = (Stage) sign_in_button.getScene().getWindow();
    stage.setScene(new SeatSelectionPage());
  }
}
