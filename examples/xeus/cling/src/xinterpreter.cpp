#include "nlohmann/json.hpp"
#include "xeus/xhelper.hpp"

#include "xinterpreter.hpp"

namespace xcling {
xinterpreter::xinterpreter(int argc, const char *const *argv, const char *llvm_dir)
    : m_cling(argc, argv, llvm_dir) {}

void xinterpreter::configure_impl() {
  //
}

void xinterpreter::execute_request_impl(xeus::xinterpreter::send_reply_callback cb,
                                        int execution_counter, const std::string &code,
                                        xeus::execute_request_config config,
                                        nl::json user_expressions) {
  std::string ename;
  std::string evalue;
  cling::Value output;
  cling::Interpreter::CompilationResult result;
  try {
    result = m_cling.process(code, &output, nullptr, true);
  } catch (cling::InterpreterException &e) {
    ename = "Interpreter Exception";
    evalue = e.what();
  } catch (std::exception &e) {
    ename = "Standard Exception";
    evalue = e.what();
  } catch (...) {
    ename = "Unknown Exception";
  }

  if (result != cling::Interpreter::kSuccess) {
    ename = "Interpreter Error";
  }

  if (!ename.empty()) {
    std::vector<std::string> traceback({ename + ": " + evalue});
    publish_execution_error(ename, evalue, traceback);
    cb(xeus::create_error_reply());
    return;
  }

  if (!output.isValid()) {
    nl::json data;
    data["text/plain"] = "Invalid Output";
    publish_execution_result(execution_counter, data, nl::json::object());
    cb(xeus::create_successful_reply());
    return;
  }

  std::string output_str;
  llvm::raw_string_ostream os(output_str);
  output.print(os);

  nl::json data;
  data["text/plain"] = output_str;
  publish_execution_result(execution_counter, data, nl::json::object());


  cb(xeus::create_successful_reply());
}

nl::json xinterpreter::complete_request_impl(const std::string &code, int cursor_pos) {
  return nl::json::object();
};

nl::json xinterpreter::inspect_request_impl(const std::string &code, int cursor_pos,
                                            int detail_level) {
  return nl::json::object();
};

nl::json xinterpreter::is_complete_request_impl(const std::string &code) {
  return nl::json::object();
};

nl::json xinterpreter::kernel_info_request_impl() {
  return xeus::create_info_reply("", "xcling", "2025.7.0", "c++", "20", "text/x-python", ".py");
};

void xinterpreter::shutdown_request_impl() {};
} // namespace xcling
