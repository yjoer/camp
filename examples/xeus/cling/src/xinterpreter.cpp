#include <regex>

#include "boost/tokenizer.hpp"
#include "nlohmann/json.hpp"
#include "xeus/xhelper.hpp"

#include "xinterpreter.hpp"

namespace xcling {
xinterpreter::xinterpreter(int argc, const char *const *argv, const char *llvm_dir)
    : m_cling(argc, argv, llvm_dir),
      m_cout_buff("stdout", std::cout.rdbuf(), true,
                  std::bind(&xeus::xinterpreter::publish_stream, this, std::placeholders::_1,
                            std::placeholders::_2)),
      m_cerr_buff("stderr", std::cerr.rdbuf(), true,
                  std::bind(&xeus::xinterpreter::publish_stream, this, std::placeholders::_1,
                            std::placeholders::_2)) {
  m_cout_sbuff = std::cout.rdbuf();
  m_cerr_sbuff = std::cerr.rdbuf();
  std::cout.rdbuf(&m_cout_buff);
  std::cerr.rdbuf(&m_cerr_buff);
}

void xinterpreter::configure_impl() {}

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
    data["text/plain"] = "(invalid)";
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
  std::size_t _cursor_pos = cursor_pos;
  std::vector<std::string> completions;
  cling::Interpreter::CompilationResult result;

  result = m_cling.codeComplete(code, _cursor_pos, completions);

  boost::tokenizer<> tok(code.substr(0, cursor_pos + 1));
  std::string last_token;
  for (boost::tokenizer<>::iterator begin = tok.begin(); begin != tok.end(); ++begin) {
    last_token = *begin;
  }

  for (auto &c : completions) {
    // remove the type at the beginning, [#int#]
    c = std::regex_replace(c, std::regex("\\[\\#.*\\#\\]"), "");

    // remove the variable name in <#type name#>
    c = std::regex_replace(c, std::regex("(\\ |\\*)+(\\w+)(\\#\\>)"), "$1$3");
    // remove unnecessary space at the end of <#type   #>
    c = std::regex_replace(c, std::regex("\\ *(\\#\\>)"), "$1");
    // remove <# #> to keep only the type
    c = std::regex_replace(c, std::regex("\\<\\#([^#>]*)\\#\\>"), "$1");
  }

  return xeus::create_complete_reply(completions, cursor_pos - last_token.length(), cursor_pos,
                                     nl::json::object());
};

nl::json xinterpreter::inspect_request_impl(const std::string &code, int cursor_pos,
                                            int detail_level) {
  return nl::json::object();
};

nl::json xinterpreter::is_complete_request_impl(const std::string &code) {
  return nl::json::object();
};

nl::json xinterpreter::kernel_info_request_impl() {
  return xeus::create_info_reply("", "xcling", "2025.7.0", "c++", "20", "text/x-c++src", ".cpp", "",
                                 "text/x-c++src");
};

void xinterpreter::shutdown_request_impl() {};
} // namespace xcling
