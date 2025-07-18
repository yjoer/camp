#include <iostream>

#include "xeus-zmq/xserver_zmq.hpp"
#include "xeus-zmq/xzmq_context.hpp"
#include "xeus/xkernel.hpp"
#include "xeus/xkernel_configuration.hpp"

#include "xinterpreter.hpp"

using interpreter_ptr = std::unique_ptr<xcling::interpreter>;
interpreter_ptr create_interpreter();

int main(int argc, char *argv[]) {
  std::unique_ptr<xeus::xcontext> context = xeus::make_zmq_context();

  interpreter_ptr interpreter = create_interpreter();

  if (argc > 1) {
    std::string filename = argv[2];
    xeus::xconfiguration config = xeus::load_configuration(filename);

    xeus::xkernel kernel(config, xeus::get_user_name(), std::move(context), std::move(interpreter),
                         xeus::make_xserver_default);

    std::cout << "Starting xcling kernel...\n\n"
              << "If you want to connect to this kernel from another client, you can use the "
              << filename + "file.\n";

    kernel.start();

    return 0;
  }

  xeus::xkernel kernel(xeus::get_user_name(), std::move(context), std::move(interpreter),
                       xeus::make_xserver_default);
  const auto &config = kernel.get_config();

  std::cout << "Starting xcling kernel...\n\n"
            << "If you want to connect to this kernel from another client, just copy as and the "
            << "following content inside a `kernel.json` file. And then run for example:\n\n"
            << "# jupyter console --existing kernel.json\n\n"
            << "kernel.json\n"
            << "```\n"
            << "{\n"
            << "    \"transport\": \"" + config.m_transport + "\",\n"
            << "    \"ip\": \"" + config.m_ip + "\",\n"
            << "    \"control_port\": " + config.m_control_port + ",\n"
            << "    \"shell_port\": " + config.m_shell_port + ",\n"
            << "    \"stdin_port\": " + config.m_stdin_port + ",\n"
            << "    \"iopub_port\": " + config.m_iopub_port + ",\n"
            << "    \"hb_port\": " + config.m_hb_port + ",\n"
            << "    \"signature_scheme\": \"" + config.m_signature_scheme + "\",\n"
            << "    \"key\": \"" + config.m_key + "\"\n"
            << "}\n"
            << "```\n";

  kernel.start();

  return 0;
}

interpreter_ptr create_interpreter() {
  int argc = 3;
  const char **argv = new const char *[argc];
  argv[0] = "xcling";

  std::string std = "-std=c++20";
  argv[1] = std.c_str();

  std::string include_dir = "-I" + std::string(CLING_ROOT_DIR) + "/include";
  argv[2] = include_dir.c_str();

  interpreter_ptr interpreter = interpreter_ptr(new xcling::interpreter(argc, argv));
  delete[] argv;
  return interpreter;
}
