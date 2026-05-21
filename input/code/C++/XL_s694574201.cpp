// This is free and unencumbered software released into the public domain.

// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.

// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

// For more information, please refer to <http://unlicense.org>

/****************/
/* template.hpp */
/****************/

#include <algorithm>
#include <cassert>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

using std::abs;
using std::cerr;
using std::cout;
using std::endl;
using std::max;
using std::min;
using std::numeric_limits;
using std::swap;

struct BoolName : std::numpunct<char> {
  std::string t, f;
  BoolName (std::string t, std::string f) : t(t), f(f) {}
  std::string do_truename() const {return t;}
  std::string do_falsename() const {return f;}
};

void set_bool_name(std::string t, std::string f) {
  cout.imbue(std::locale(cout.getloc(), new BoolName(t, f)));
}

struct Initializer {
  Initializer() {
    cout << std::fixed << std::setprecision(15) << std::boolalpha;
    set_bool_name("Yes", "No");
  }
} initializer;

struct Input {
  bool eof;

  Input() : eof(false) {}

  operator char() {
    char v;
    this->eof = (std::scanf("%c", &v) != 1);
    return v;
  }

  operator int() {
    int v;
    this->eof = (std::scanf("%d", &v) != 1);
    return v;
  }

  operator long() {
    long v;
    this->eof = (std::scanf("%ld", &v) != 1);
    return v;
  }

  operator long long() {
    long long v;
    this->eof = (std::scanf("%lld", &v) != 1);
    return v;
  }

  operator double() {
    double v;
    this->eof = (std::scanf("%lf", &v) != 1);
    return v;
  }

  operator long double() {
    long double v;
    this->eof = (std::scanf("%Lf", &v) != 1);
    return v;
  }

  void ignore() const {
    getchar();
  }
} in;

template<typename T> bool chmin(T& a, T b) {return a > b ? a = b, true : false;}

template<typename T> bool chmax(T& a, T b) {return a < b ? a = b, true : false;}

template<typename T, typename S> std::function<S(T)> cast() {
  return [](const T& t) { return static_cast<S>(t); };
}

/******************/
/* arithmetic.hpp */
/******************/

template<typename T> class Addition {
public:
  template<typename V> T operator+(const V& v) const {
    return T(static_cast<const T&>(*this)) += v;
  }

  T operator++() {
    return static_cast<T&>(*this) += 1;
  }
};

template<typename T> class Subtraction {
public:
  template<typename V> T operator-(const V& v) const {
    return T(static_cast<const T&>(*this)) -= v;
  }
};

template<typename T> class Multiplication {
public:
  template<typename V> T operator*(const V& v) const {
    return T(static_cast<const T&>(*this)) *= v;
  }
};

template<typename T> class Division {
public:
  template<typename V> T operator/(const V& v) const {
    return T(static_cast<const T&>(*this)) /= v;
  }
};

template<typename T> class Modulus {
public:
  template<typename V> T operator%(const V& v) const {
    return T(static_cast<const T&>(*this)) %= v;
  }
};

template<typename T> class IndivisibleArithmetic : public Addition<T>, public Subtraction<T>, public Multiplication<T> {};

template<typename T> class Arithmetic : public IndivisibleArithmetic<T>, public Division<T> {};

/*****************/
/* container.hpp */
/*****************/

#include <vector>

template<typename T> class Container : public T {
private:
  using S = typename T::value_type;

public:
  Container() : T() {}

  Container(int n) : T(n) {}

  Container(int n, S s) : T(n, s) {}

  template<typename Itr> Container(Itr first, Itr last) : T(first, last) {}

  Container(const std::initializer_list<S>& v) : T(v) {}

  Container(int n, Input& in) {
    std::vector<S> v(n);
    for (auto& i : v) {
      i = in;
    }
    *this = Container<T>(v.begin(), v.end());
  }

  bool in(const S& a) const {
    return find(this->begin(), this->end(), a) != this->end();
  }
};

/***************/
/* ordered.hpp */
/***************/

template<typename T> class Ordered {
public:
  template<typename V> bool operator==(const V& v) const {
    return !(static_cast<T>(v) < static_cast<const T&>(*this) || static_cast<const T&>(*this) < static_cast<T>(v));
  }
  
  template<typename V> bool operator!=(const V& v) const {
    return static_cast<T>(v) < static_cast<const T&>(*this) || static_cast<const T&>(*this) < static_cast<T>(v);
  }

  template<typename V> bool operator>(const V& v) const {
    return static_cast<T>(v) < static_cast<const T&>(*this);
  }

  template<typename V> bool operator<=(const V& v) const {
    return !(static_cast<T>(v) < static_cast<const T&>(*this));
  }

  template<typename V> bool operator>=(const V& v) const {
    return !(static_cast<const T&>(*this) < static_cast<T>(v));
  }
};

/**************/
/* vector.hpp */
/**************/

template<typename T> class Vector : public Container<std::vector<T>>, public Arithmetic<Vector<T>>, public Modulus<Vector<T>>, public Ordered<Vector<T>> {
public:
  Vector() = default;

  Vector(const Vector<T>& v) = default;

  Vector(int n) : Container<std::vector<T>>(n) {}

  Vector(int n, T t) : Container<std::vector<T>>(n, t) {}

  template<typename Itr> Vector(Itr first, Itr last) : Container<std::vector<T>>(first, last) {}

  Vector(const std::initializer_list<T>& v) : Container<std::vector<T>>(v) {}

  Vector(int n, Input& in) : Container<std::vector<T>>(n, in) {}

  Vector operator+=(const Vector& v) {
    for (unsigned i = 0; i < this->size(); ++i) (*this)[i] += v[i];
    return *this;
  }

  Vector operator+=(const T& v) {
    for (auto& i : *this) {
      i += v;
    }
    return *this;
  }

  Vector operator-=(const Vector& v) {
    for (unsigned i = 0; i < this->size(); ++i) (*this)[i] -= v[i];
    return *this;
  }

  Vector operator*=(const Vector& v) {
    for (unsigned i = 0; i < this->size(); ++i) (*this)[i] *= v[i];
    return *this;
  }

  Vector operator/=(const Vector& v) {
    for (unsigned i = 0; i < this->size(); ++i) (*this)[i] /= v[i];
    return *this;
  }

  Vector operator%=(const Vector& v) {
    for (unsigned i = 0; i < this->size(); ++i) (*this)[i] %= v[i];
    return *this;
  }

  bool operator<(const Vector& v) const {
    if (this->size() != v.size()) return this->size() < v.size();
    for (unsigned i = 0; i < this->size(); ++i) {
      if ((*this)[i] != v[i]) {
        return (*this)[i] < v[i];
      }
    }
    return false;
  }

  T inner_product(const Vector<T>& v) const {
    return std::inner_product(this->begin(), this->end(), v.begin(), T(0));
  }

  void output(char sep = '\n', char end = '\n') const {
    if (!this->empty()) {
      cout << (*this)[0];
    }
    for (unsigned i = 1; i < this->size(); ++i) {
      cout << sep << (*this)[i];
    }
    cout << end;
  }

  Vector<T> partial_sort(int k, bool reverse = false) {
    if (!reverse) {
      std::partial_sort(this->begin(), this->begin() + k, this->end());
    } else {
      std::partial_sort(this->begin(), this->begin() + k, this->end(), std::greater<T>());
    }
    return *this;
  }

  Vector<T> sort() {
    std::sort(this->begin(), this->end());
    return *this;
  }

  Vector<T> rsort() {
    std::sort(this->rbegin(), this->rend());
    return *this;
  }

  Vector<T> subvector(int a) const {
    return Vector<T>(this->begin(), this->begin() + a);
  }

  Vector<T> subvector(int a, int b) const {
    return Vector<T>(this->begin() + a, this->begin() + b);
  }

  template<typename Function> auto transform(Function func) const -> Vector<decltype(func(T()))> {
    Vector<decltype(func(T()))> res;
    std::transform(this->begin(), this->end(), std::back_inserter(res), func);
    return res;
  }

  Vector<T> partial_sum() const {
    Vector<T> res;
    std::partial_sum(this->begin(), this->end(), std::back_inserter(res));
    return res;
  }

  Vector<T> reverse() {
    std::reverse(this->begin(), this->end());
    return *this;
  }

  template<typename Function> bool all_of(Function func) {
    return std::all_of(this->begin(), this->end(), func);
  }

  Vector<T> adjacent_difference() const {
    Vector<T> res;
    std::adjacent_difference(this->begin(), this->end(), std::back_inserter(res));
    return res;
  }

  T max() const {
    return *max_element(this->begin(), this->end());
  }

  T min() const {
    return *min_element(this->begin(), this->end());
  }

  int argmax() const {
    return max_element(this->begin(), this->end()) - this->begin();
  }

  int argmin() const {
    return min_element(this->begin(), this->end()) - this->begin();
  }
};

template<typename T> Vector<T> iota(int n) {
  Vector<T> v(n);
  std::iota(v.begin(), v.end(), T(0));
  return v;
}

/*************/
/* stack.hpp */
/*************/

#include <stack>

template<typename T> class Stack : public std::stack<T, std::vector<T>> {
public:
  Stack() : std::stack<T, std::vector<T>>() {}

  Stack(const std::initializer_list<T>& s) : std::stack<T, std::vector<T>>(s) {}

  T top() {
    T res = std::stack<T, std::vector<T>>::top();
    this->pop();
    return res;
  }

  T peek() const {
    return std::stack<T, std::vector<T>>::top();
  }
};

/************/
/* main.cpp */
/************/

int main() {
  Stack<int> s;
  for (int n; n = in, !in.eof;) {
    if (n) {
      s.push(n);
    } else {
      cout << s.top() << endl;
    }
  }
}


