#ifndef XCLING_INTERPRETER_HPP
#define XCLING_INTERPRETER_HPP

#include "xeus/xinterpreter.hpp"

using namespace xeus;
using xeus::xinterpreter;

namespace xcling {
class interpreter : public xinterpreter {
public:
  interpreter() = default;
  virtual ~interpreter() = default;

private:
  void configure_impl() override;

  void execute_request_impl(send_reply_callback cb, int execution_counter, const std::string &code,
                            execute_request_config config, nl::json user_expressions) override;

  nl::json complete_request_impl(const std::string &code, int cursor_pos) override;

  nl::json inspect_request_impl(const std::string &code, int cursor_pos, int detail_level) override;

  nl::json is_complete_request_impl(const std::string &code) override;

  nl::json kernel_info_request_impl() override;

  void shutdown_request_impl() override;
};
} // namespace xcling

#endif
