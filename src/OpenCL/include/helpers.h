
#ifndef HELPERS__H
#define HELPERS__H

#include <string>
#include <vector>
#include <sstream>

std::vector<std::string> split(const std::string &s, char delim);

template <class T>
std::string vector_join( const std::vector<T>& v, const std::string& token ) {
  std::stringstream result;
  for (typename std::vector<T>::const_iterator i = v.begin(); i != v.end(); i++) {
    if ( i != v.begin() )
        result << token;
    result << *i;
  }
  return result.str();
}

std::string randUuid();

#endif // HELPERS__H
