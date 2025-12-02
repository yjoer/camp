import javafx.collections.ListChangeListener;

class TodoController {

  TodoModel model = new TodoModel();
  TodoView view = new TodoView(this::add_item, this::delete_item);

  TodoController() {
    model.items.addListener(
      (ListChangeListener<String>) change -> {
        while (change.next()) {
          if (change.wasAdded()) {
            for (String item : change.getAddedSubList()) {
              view.list.getItems().add(item);
            }
          }

          if (change.wasRemoved()) {
            for (String item : change.getRemoved()) {
              view.list.getItems().remove(item);
            }
          }
        }
      }
    );
  }

  void add_item() {
    String item = view.input.getText();
    if (item.isEmpty()) return;

    model.items.add(item);
    view.input.clear();
  }

  void delete_item() {
    int idx = view.list.getSelectionModel().getSelectedIndex();
    if (idx == -1) return;

    model.items.remove(idx);
  }

  TodoView view() {
    return view;
  }
}
