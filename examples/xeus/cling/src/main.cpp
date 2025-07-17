#include <iostream>

#include "xeus-zmq/xserver_zmq.hpp"
#include "xeus-zmq/xzmq_context.hpp"
#include "xeus/xkernel.hpp"
#include "xeus/xkernel_configuration.hpp"

#include "xinterpreter.hpp"

int main(int argc, char *argv[]) {
  std::unique_ptr<xeus::xcontext> context = xeus::make_zmq_context();

  using interpreter_ptr = std::unique_ptr<xcling::interpreter>;
  interpreter_ptr interpreter = interpreter_ptr(new xcling::interpreter());

  if (argc > 1) {
    std::string filename = argc == 1 ? "connection.json" : argv[2];
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
