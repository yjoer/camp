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

	TodoViewModel todo_vm = new TodoViewModel();

	TodoView() {
		Text title = new Text("Todo MVVM");

		TextField input = new TextField();
		VBox.setMargin(input, new Insets(4, 0, 0, 0));
		input.textProperty().bindBidirectional(todo_vm.input);

		ListView<String> list = new ListView<>();
		VBox.setMargin(list, new Insets(16, 0, 0, 0));
		list.setItems(todo_vm.items);
		todo_vm.selected_item.bind(list.getSelectionModel().selectedIndexProperty());

		Buttons buttons = new Buttons(todo_vm);

		this.setPadding(new Insets(8, 16, 8, 16));
		this.getChildren().addAll(title, input, buttons, list);
	}
}

class Buttons extends ButtonBar {

	Buttons(TodoViewModel todo_vm) {
		Button add_button = new Button("Add");
		ButtonBar.setButtonData(add_button, ButtonData.LEFT);
		add_button.setOnAction(e -> todo_vm.add_item());

		Button delete_button = new Button("Delete");
		ButtonBar.setButtonData(delete_button, ButtonData.LEFT);
		delete_button.setOnAction(e -> todo_vm.delete_item());

		Region pad = new Region();
		HBox.setHgrow(pad, Priority.ALWAYS);

		this.setPadding(new Insets(8, 0, 0, 0));
		this.getButtons().addAll(add_button, delete_button, pad);
	}
}
