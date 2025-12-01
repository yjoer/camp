import java.io.IOException;
import java.util.regex.Pattern;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import lib.DataRepository;

class SignUpPage extends Scene {

  SignUpPage() {
    super(on_create(), 800, 600);
  }

  private static Parent on_create() {
    try {
      FXMLLoader loader = new FXMLLoader(SignUpPage.class.getResource("/sign-up.fxml"));
      loader.setController(new SignUpController());

      return loader.load();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}

class SignUpController {

  @FXML
  Text message;

  @FXML
  TextField first_name;

  @FXML
  TextField last_name;

  @FXML
  TextField email_address;

  @FXML
  TextField password;

  @FXML
  TextField confirm_password;

  @FXML
  TextField phone_number;

  @FXML
  Button sign_up_button;

  @FXML
  Button cancel_button;

  @FXML
  void handle_sign_up() {
    if (first_name.getText().isEmpty()) {
      message.setText("First name is required.");
      return;
    }

    if (last_name.getText().isEmpty()) {
      message.setText("Last name is required.");
      return;
    }

    if (email_address.getText().isEmpty()) {
      message.setText("Email address is required.");
      return;
    }

    Pattern p = Pattern.compile("^[a-z0-9+-._]+@[a-z0-9.-]+$", Pattern.CASE_INSENSITIVE);
    if (!p.matcher(email_address.getText()).matches()) {
      message.setText("Email address is invalid.");
      return;
    }

    if (password.getText().isEmpty()) {
      message.setText("Password is required.");
      return;
    }

    if (!password.getText().equals(confirm_password.getText())) {
      message.setText("Those passwords didn't match. Try again.");
      return;
    }

    if (phone_number.getText().isEmpty()) {
      message.setText("Phone number is required.");
      return;
    }

    DataRepository repo = new DataRepository();
    String[] parts = email_address.getText().split("@");
    repo.create_user(
      first_name.getText(),
      last_name.getText(),
      email_address.getText(),
      password.getText(),
      phone_number.getText(),
      parts[1].equals("usm.my") ? "administrator" : "student"
    );

    Stage stage = (Stage) sign_up_button.getScene().getWindow();
    stage.setScene(new SignInPage());
  }

  @FXML
  void handle_cancel() {
    Stage stage = (Stage) cancel_button.getScene().getWindow();
    stage.setScene(new SignInPage());
  }
}
