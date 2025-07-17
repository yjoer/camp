#include "nlohmann/json.hpp"
#include "xeus/xhelper.hpp"

#include "xinterpreter.hpp"

namespace xcling {
void interpreter::configure_impl() {
  //
}

void interpreter::execute_request_impl(send_reply_callback cb, int execution_counter,
                                       const std::string &code, execute_request_config config,
                                       nl::json user_expressions) {
  nl::json pub_data;
  pub_data["text/plain"] = "hello world!";
  publish_execution_result(execution_counter, std::move(pub_data), nl::json::object());
  publish_execution_error("TypeError", "123", {"!@$!@", "*(*"});

  cb(xeus::create_successful_reply());
}
} // namespace xcling
