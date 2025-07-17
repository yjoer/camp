#ifndef XCLING_INTERPRETER_HPP
#define XCLING_INTERPRETER_HPP

#include "cling/Interpreter/Exception.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"
#include "xeus/xinterpreter.hpp"
#include "llvm/Support/raw_ostream.h"

#include "xbuffer.hpp"

namespace xcling {
class xinterpreter : public xeus::xinterpreter {
public:
  xinterpreter(int argc, const char *const *argv, const char *llvm_dir);
  virtual ~xinterpreter() = default;

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

  cling::Interpreter m_cling;

  std::streambuf *m_cout_sbuff;
  std::streambuf *m_cerr_sbuff;
  xoutput_buffer m_cout_buff;
  xoutput_buffer m_cerr_buff;
};
} // namespace xcling

#endif
