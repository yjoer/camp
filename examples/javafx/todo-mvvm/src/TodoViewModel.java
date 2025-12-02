import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;

class TodoViewModel {

  StringProperty input = new SimpleStringProperty("");

  ObservableList<String> items = FXCollections.observableArrayList();
  IntegerProperty selected_item = new SimpleIntegerProperty(-1);

  void add_item() {
    String item = input.get();
    if (item.isEmpty()) return;

    items.add(item);
    input.set("");
  }

  void delete_item() {
    int idx = selected_item.get();
    if (idx == -1) return;

    items.remove(idx);
  }
}
