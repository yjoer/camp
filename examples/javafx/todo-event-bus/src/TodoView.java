import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonBar;
import javafx.scene.control.ListView;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;

class TodoView extends VBox {

  TextField input = new TextField();
  ListView<String> list = new ListView<>();

  TodoView() {
    TodoEventBus.get().register(this);

    Text title = new Text("Todo EventBus");

    VBox.setMargin(input, new Insets(4, 0, 0, 0));
    VBox.setMargin(list, new Insets(16, 0, 0, 0));

    Buttons buttons = new Buttons();

    this.setPadding(new Insets(8, 16, 8, 16));
    this.getChildren().addAll(title, input, buttons, list);
  }

  @Subscribe
  void handle_add(AddEvent event) {
    String item = input.getText();
    if (item.isEmpty()) return;

    list.getItems().add(item);
    input.clear();
  }

  @Subscribe
  void handle_delete(DeleteEvent event) {
    int idx = list.getSelectionModel().getSelectedIndex();
    if (idx == -1) return;

    list.getItems().remove(idx);
  }
}

class Buttons extends ButtonBar {

  Buttons() {
    EventBus bus = TodoEventBus.get();

    Button add_button = new Button("Add");
    ButtonBar.setButtonData(add_button, ButtonData.LEFT);
    add_button.setOnAction(e -> bus.post(new AddEvent()));

    Button delete_button = new Button("Delete");
    ButtonBar.setButtonData(delete_button, ButtonData.LEFT);
    delete_button.setOnAction(e -> bus.post(new DeleteEvent()));

    Region pad = new Region();
    HBox.setHgrow(pad, Priority.ALWAYS);

    this.setPadding(new Insets(8, 0, 0, 0));
    this.getButtons().addAll(add_button, delete_button, pad);
  }
}
