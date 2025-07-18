#include <iostream>

#include "boost/process/environment.hpp"
#include "xeus-zmq/xserver_zmq.hpp"
#include "xeus-zmq/xzmq_context.hpp"
#include "xeus/xkernel.hpp"
#include "xeus/xkernel_configuration.hpp"

#include "xinterpreter.hpp"

using namespace boost::process;

using interpreter_ptr = std::unique_ptr<xcling::xinterpreter>;
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

  auto cling_path = environment::find_executable("cling");
  if (cling_path.empty()) {
    std::cerr << "Error: Could not find the cling executable in the PATH.\n";
    exit(EXIT_FAILURE);
  }

  auto cling_root_path = cling_path.parent_path().parent_path();

  std::string include_dir = (cling_root_path / "include").string();
  std::string include_path = "-I" + include_dir;
  argv[2] = include_path.c_str();

  std::string cling_root_dir = cling_root_path.string();
  std::cout << "Cling executable detected.\n"
            << "Root directory for Cling: " << cling_root_dir << "\n"
            << "Using standard: " << std << "\n"
            << "Including headers from: " << include_dir << "\n\n";

  interpreter_ptr interpreter =
      interpreter_ptr(new xcling::xinterpreter(argc, argv, cling_root_dir.c_str()));

  delete[] argv;
  return interpreter;
}
