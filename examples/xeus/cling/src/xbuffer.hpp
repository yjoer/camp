#ifndef XCLING_BUFFER_HPP
#define XCLING_BUFFER_HPP

#include <functional>
#include <mutex>
#include <streambuf>

namespace xcling {
class xoutput_buffer : public std::streambuf {
public:
  using callback_type = std::function<void(const std::string &, const std::string &)>;

  xoutput_buffer(std::string name, std::streambuf *buff, bool tee, callback_type callback)
      : m_name(name), m_buff(buff), m_tee(tee), m_callback(std::move(callback)) {};

protected:
  traits_type::int_type overflow(traits_type::int_type c) override {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (traits_type::eq_int_type(c, traits_type::eof()))
      return c;

    m_buffer.push_back(traits_type::to_char_type(c));

    if (m_tee)
      m_buff->sputc(c);

    return c;
  }

  std::streamsize xsputn(const char *s, std::streamsize count) override {
    std::lock_guard<std::mutex> lock(m_mutex);

    m_buffer.append(s, count);

    if (m_tee)
      m_buff->sputn(s, count);

    return count;
  };

  traits_type::int_type sync() override {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_buffer.empty())
      return 0;

    m_callback(m_name, m_buffer);
    m_buffer.clear();

    if (m_tee)
      m_buff->pubsync();

    return 0;
  }

  std::string m_name;
  std::streambuf *m_buff;
  bool m_tee;
  callback_type m_callback;

  std::mutex m_mutex;
  std::string m_buffer;
};
} // namespace xcling

#endif
