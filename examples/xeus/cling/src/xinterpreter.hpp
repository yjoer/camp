#ifndef XCLING_INTERPRETER_HPP
#define XCLING_INTERPRETER_HPP

#include "cling/Interpreter/Interpreter.h"
#include "xeus/xinterpreter.hpp"

namespace xcling {
class interpreter : public xeus::xinterpreter {
public:
  interpreter(int argc, const char *const *argv);
  virtual ~interpreter() = default;

private:
  void configure_impl() override;

  void execute_request_impl(xeus::xinterpreter::send_reply_callback cb, int execution_counter,
                            const std::string &code, xeus::execute_request_config config,
                            nl::json user_expressions) override;

  nl::json complete_request_impl(const std::string &code, int cursor_pos) override;

  nl::json inspect_request_impl(const std::string &code, int cursor_pos, int detail_level) override;

  nl::json is_complete_request_impl(const std::string &code) override;

  nl::json kernel_info_request_impl() override;

  void shutdown_request_impl() override;

  cling::Interpreter m_interpreter;
};
} // namespace xcling

#endif
