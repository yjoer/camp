package lib;

public class AppState {

  static AppState ref = null;

  public static AppState get_instance() {
    if (ref == null) ref = new AppState();
    return ref;
  }

  int user_id;

  public int user_id() {
    return user_id;
  }

  public void set_user_id(int user_id) {
    this.user_id = user_id;
  }
}
