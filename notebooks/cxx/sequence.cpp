// %%
gClingOpts->AllowRedefinition = 1;

// %%
#include <iostream>
#include <optional>

// %% [markdown]
// ## Linked List

// %%
template <class T> struct forward_list_node {
  T data;
  forward_list_node<T> *next;
};

// %%
template <class T> class forward_list {
protected:
  forward_list_node<T> *head;
  forward_list_node<T> *tail;

public:
  forward_list();

  std::optional<T *> front();
  bool empty();
  void clear();
  void push_front(const T &value);
  void pop_front();
  void reverse();
};

// %%
template <class T> forward_list<T>::forward_list() : head(nullptr), tail(nullptr) {}

// %% [markdown]
// ### Front

// %%
template <class T> std::optional<T *> forward_list<T>::front() {
  if (head == nullptr)
    return std::nullopt;

  return &head->data;
}

// %% [markdown]
// ### Empty

// %%
template <class T> bool forward_list<T>::empty() {
  return head == nullptr;
}

// %% [markdown]
// ### Clear

// %%
template <class T> void forward_list<T>::clear() {
  forward_list_node<T> *temp;
  while (head != nullptr) {
    temp = head;
    head = head->next;
    delete temp;
  }

  tail = nullptr;
}

// %% [markdown]
// ### Push Front

// %%
template <class T> void forward_list<T>::push_front(const T &value) {
  forward_list_node<T> *node = new forward_list_node<T>;
  node->data = value;
  node->next = head;

  head = node;

  if (tail == nullptr)
    tail = node;
}

// %% [markdown]
// ### Pop Front

// %%
template <class T> void forward_list<T>::pop_front() {
  if (head == nullptr)
    return;

  forward_list_node<T> *temp = head;
  head = head->next;
  delete temp;
}

// %% [markdown]
// ### Reverse

// %%
template <class T> void forward_list<T>::reverse() {
  tail = head;

  forward_list_node<T> *prev = nullptr;
  forward_list_node<T> *curr = head;
  while (curr != nullptr) {
    forward_list_node<T> *next = curr->next;
    curr->next = prev;
    prev = curr;
    curr = next;
  }

  head = prev;
}

// %% [markdown]
// ### Usage

// %%
forward_list<int> fwl;
fwl.push_front(1);
fwl.push_front(2);
fwl.pop_front();
std::cout << **fwl.front() << std::endl;
fwl.push_front(3);
fwl.push_front(4);
fwl.clear();
fwl.push_front(5);
std::cout << **fwl.front() << std::endl;
fwl.clear();
std::cout << fwl.empty();

// %%
fwl.push_front(1);
fwl.push_front(2);
fwl.push_front(3);
fwl.reverse();
std::cout << **fwl.front() << std::endl;
fwl.pop_front();
std::cout << **fwl.front() << std::endl;
fwl.pop_front();
std::cout << **fwl.front() << std::endl;
fwl.pop_front();
std::cout << fwl.empty();

// %% [markdown]
// ## Doubly Linked List

// %%
template <class T> struct list_node {
  T data;
  list_node<T> *prev;
  list_node<T> *next;
};

// %%
template <class T> class list {
protected:
  list_node<T> *head;
  list_node<T> *tail;

public:
  list();

  std::optional<T *> front();
  std::optional<T *> back();
  bool empty();
  void clear();
  void push_front(const T &value);
  void push_back(const T &value);
  void pop_front();
  void pop_back();
  void reverse();
};

// %%
template <class T> list<T>::list() : head(nullptr), tail(nullptr) {}

// %% [markdown]
// ### Front

// %%
template <class T> std::optional<T *> list<T>::front() {
  if (head == nullptr)
    return std::nullopt;

  return &head->data;
}

// %% [markdown]
// ### Back

// %%
template <class T> std::optional<T *> list<T>::back() {
  if (tail == nullptr)
    return std::nullopt;

  return &tail->data;
}

// %% [markdown]
// ### Empty

// %%
template <class T> bool list<T>::empty() {
  return head == nullptr;
}

// %% [markdown]
// ### Clear

// %%
template <class T> void list<T>::clear() {
  list_node<T> *temp;
  while (head != nullptr) {
    temp = head;
    head = head->next;
    delete temp;
  }

  tail = nullptr;
}

// %% [markdown]
// ### Push Front

// %%
template <class T> void list<T>::push_front(const T &value) {
  list_node<T> *node = new list_node<T>;
  node->data = value;
  node->prev = nullptr;
  node->next = head;

  if (head != nullptr)
    head->prev = node;

  head = node;

  if (tail == nullptr)
    tail = node;
}

// %% [markdown]
// ### Push Back

// %%
template <class T> void list<T>::push_back(const T &value) {
  list_node<T> *node = new list_node<T>;
  node->data = value;
  node->prev = tail;
  node->next = nullptr;

  if (tail != nullptr)
    tail->next = node;

  tail = node;

  if (head == nullptr)
    head = node;
}

// %% [markdown]
// ### Pop Front

// %%
template <class T> void list<T>::pop_front() {
  if (head == nullptr)
    return;

  list_node<T> *temp = head;
  head = head->next;
  delete temp;

  if (head != nullptr) {
    head->prev = nullptr;
  } else {
    tail = nullptr;
  }
}

// %% [markdown]
// ### Pop Back

// %%
template <class T> void list<T>::pop_back() {
  if (tail == nullptr)
    return;

  list_node<T> *temp = tail;
  tail = tail->prev;
  delete temp;

  if (tail != nullptr) {
    tail->next = nullptr;
  } else {
    head = nullptr;
  }
}

// %% [markdown]
// ### Reverse

// %%
template <class T> void list<T>::reverse() {
  list_node<T> *prev = nullptr;
  list_node<T> *curr = tail;
  while (curr != nullptr) {
    prev = curr->prev;
    curr->prev = curr->next;
    curr->next = prev;
    curr = curr->next;
  }

  list_node<T> *temp = head;
  head = tail;
  tail = temp;
}

// %% [markdown]
// ### Usage

// %%
list<int> l;
l.push_front(2);
l.push_front(1);
l.push_back(3);
std::cout << **l.front() << std::endl;
std::cout << **l.back() << std::endl;
l.pop_front();
l.pop_back();
std::cout << **l.front() << std::endl;
l.push_front(4);
l.clear();
l.push_front(5);
std::cout << **l.back() << std::endl;
l.clear();
std::cout << l.empty();

// %%
l.push_front(1);
l.push_front(2);
l.push_front(3);
l.reverse();
std::cout << **l.front() << std::endl;
l.pop_front();
std::cout << **l.front() << std::endl;
l.pop_front();
std::cout << **l.front() << std::endl;
l.pop_front();
std::cout << l.empty();

// %%
l.push_back(4);
l.push_back(5);
l.push_back(6);
l.reverse();
std::cout << **l.back() << std::endl;
l.pop_back();
std::cout << **l.back() << std::endl;
l.pop_back();
std::cout << **l.back() << std::endl;
l.pop_back();
std::cout << l.empty();

// %%
