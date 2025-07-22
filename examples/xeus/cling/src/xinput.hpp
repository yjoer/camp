#ifndef XCLING_INPUT_HPP
#define XCLING_INPUT_HPP

#include <iostream>
#include <streambuf>

#include "xeus/xinput.hpp"

#include "xbuffer.hpp"

namespace xcling {
class xinput {
public:
  xinput()
      : m_cin_sbuff(std::cin.rdbuf()),
        m_cin_buff([](std::string &value) { value = xeus::blocking_input_request("", false); }) {
    std::cin.rdbuf(&m_cin_buff);
  }

  ~xinput() {
    std::cin.rdbuf(m_cin_sbuff); //
  }

protected:
  std::streambuf *m_cin_sbuff;
  xinput_buffer m_cin_buff;
};
} // namespace xcling

#endif
