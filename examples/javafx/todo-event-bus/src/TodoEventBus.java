import com.google.common.eventbus.EventBus;

class TodoEventBus {

  static EventBus ref = null;

  static EventBus get() {
    if (ref == null) ref = new EventBus();
    return ref;
  }
}

record AddEvent() {}

record DeleteEvent() {}
