#line 1 "main.cpp"

/**
 * @title Template
 */

#include <iostream>
#include <algorithm>
#include <utility>
#include <numeric>
#include <vector>
#include <array>
#include <cassert>

#line 2 "/Users/kodamankod/Desktop/cpp_programming/Library/other/chmin_chmax.cpp"

template <class T, class U>
constexpr bool chmin(T &lhs, const U &rhs) {
  if (lhs > rhs) { lhs = rhs; return true; }
  return false;
}

template <class T, class U>
constexpr bool chmax(T &lhs, const U &rhs) {
  if (lhs < rhs) { lhs = rhs; return true; }
  return false;
}

/**
 * @title Chmin/Chmax
 */
#line 2 "/Users/kodamankod/Desktop/cpp_programming/Library/other/range.cpp"

#line 4 "/Users/kodamankod/Desktop/cpp_programming/Library/other/range.cpp"

class range {
public:
  class iterator {
  private:
    int64_t M_position;

  public:
    constexpr iterator(int64_t position) noexcept: M_position(position) { }
    constexpr void operator ++ () noexcept { ++M_position; }
    constexpr bool operator != (iterator other) const noexcept { return M_position != other.M_position; }
    constexpr int64_t operator * () const noexcept { return M_position; }
  };

  class reverse_iterator {
  private:
    int64_t M_position;
  
  public:
    constexpr reverse_iterator(int64_t position) noexcept: M_position(position) { }
    constexpr void operator ++ () noexcept { --M_position; }
    constexpr bool operator != (reverse_iterator other) const noexcept { return M_position != other.M_position; }
    constexpr int64_t operator * () const noexcept { return M_position; }
  };
  
private:
  const iterator M_first, M_last;

public:
  constexpr range(int64_t first, int64_t last) noexcept: M_first(first), M_last(std::max(first, last)) { }
  constexpr iterator begin() const noexcept { return M_first; }
  constexpr iterator end() const noexcept { return M_last; }
  constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(*M_last - 1); } 
  constexpr reverse_iterator rend() const noexcept { return reverse_iterator(*M_first - 1); } 
};

/**
 * @title Range
 */
#line 2 "/Users/kodamankod/Desktop/cpp_programming/Library/other/rev.cpp"

#include <type_traits>
#include <iterator>
#line 6 "/Users/kodamankod/Desktop/cpp_programming/Library/other/rev.cpp"

template <class T>
class rev_impl {
public:
  using iterator = decltype(std::rbegin(std::declval<T>()));

private:
  const iterator M_begin;
  const iterator M_end;

public:
  constexpr rev_impl(T &&cont) noexcept: M_begin(std::rbegin(cont)), M_end(std::rend(cont)) { }
  constexpr iterator begin() const noexcept { return M_begin; }
  constexpr iterator end() const noexcept { return M_end; }
};

template <class T>
constexpr decltype(auto) rev(T &&cont) {
  return rev_impl<T>(std::forward<T>(cont));
}

/**
 * @title Reverser
 */
#line 2 "/Users/kodamankod/Desktop/cpp_programming/Library/container/fenwick_tree.cpp"

#line 2 "/Users/kodamankod/Desktop/cpp_programming/Library/other/bit_operation.cpp"

#include <cstddef>
#include <cstdint>

constexpr size_t bit_ppc(const uint64_t x) { return __builtin_popcountll(x); }
constexpr size_t bit_ctzr(const uint64_t x) { return x == 0 ? 64 : __builtin_ctzll(x); }
constexpr size_t bit_ctzl(const uint64_t x) { return x == 0 ? 64 : __builtin_clzll(x); }
constexpr size_t bit_width(const uint64_t x) { return 64 - bit_ctzl(x); }
constexpr uint64_t bit_msb(const uint64_t x) { return x == 0 ? 0 : uint64_t(1) << (bit_width(x) - 1); }
constexpr uint64_t bit_lsb(const uint64_t x) { return x & (-x); }
constexpr uint64_t bit_cover(const uint64_t x) { return x == 0 ? 0 : bit_msb(2 * x - 1); }

constexpr uint64_t bit_rev(uint64_t x) {
  x = ((x >> 1) & 0x5555555555555555) | ((x & 0x5555555555555555) << 1);
  x = ((x >> 2) & 0x3333333333333333) | ((x & 0x3333333333333333) << 2);
  x = ((x >> 4) & 0x0F0F0F0F0F0F0F0F) | ((x & 0x0F0F0F0F0F0F0F0F) << 4);
  x = ((x >> 8) & 0x00FF00FF00FF00FF) | ((x & 0x00FF00FF00FF00FF) << 8);
  x = ((x >> 16) & 0x0000FFFF0000FFFF) | ((x & 0x0000FFFF0000FFFF) << 16);
  x = (x >> 32) | (x << 32);
  return x;
}

/**
 * @title Bit Operations
 */
#line 4 "/Users/kodamankod/Desktop/cpp_programming/Library/container/fenwick_tree.cpp"

#line 8 "/Users/kodamankod/Desktop/cpp_programming/Library/container/fenwick_tree.cpp"
#include <type_traits>

template <class T>
class fenwick_tree {
public:
  using value_type = T;
  using size_type = size_t;

private:
  std::vector<value_type> M_tree;

public:
  fenwick_tree() = default;
  explicit fenwick_tree(size_type size) { initialize(size); }

  void initialize(size_type size) {
    M_tree.assign(size + 1, value_type { });
  }

  void add(size_type index, const value_type& x) {
    assert(index < size());
    ++index;
    while (index <= size()) {
      M_tree[index] += x;
      index += bit_lsb(index);
    }
  }

  template <size_type Indexed = 1>
  value_type get(size_type index) const {
    assert(index < size());
    index += Indexed;
    value_type res{ };
    while (index > 0) {
      res += M_tree[index];
      index -= bit_lsb(index);
    }
    return res;
  }
  value_type fold(size_type first, size_type last) const {
    assert(first <= last);
    assert(last <= size());
    value_type res{};
    while (first < last) {
      res += M_tree[last];
      last -= bit_lsb(last);
    }
    while (last < first) {
      res -= M_tree[first];
      first -= bit_lsb(first);
    }
    return res;
  }

  template <class Func>
  size_type satisfies(const size_type left, Func &&func) const {
    assert(left <= size());
    if (func(value_type { })) return left;
    value_type val = -get<0>(left);
    size_type res = 0;
    for (size_type cur = bit_cover(size() + 1) >> 1; cur > 0; cur >>= 1) {
      if ((res + cur <= left) || (res + cur <= size() && !func(val + M_tree[res + cur]))) {
        val += M_tree[res + cur];
        res += cur;
      }
    }
    return res + 1;
  }

  void clear() {
    M_tree.clear();
    M_tree.shrink_to_fit();
  }
  size_type size() const {
    return M_tree.size() - 1;
  }
};

/**
 * @title Fenwick Tree
 */
#line 18 "main.cpp"

using i32 = int32_t;
using i64 = int64_t;
using u32 = uint32_t;
using u64 = uint64_t;

constexpr i32 inf32 = (i32(1) << 30) - 1;
constexpr i64 inf64 = (i64(1) << 62) - 1;

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  size_t N, Q;
  std::cin >> N >> Q;
  fenwick_tree<i64> fen(N);
  for (auto i: range(0, N)) {
    i32 x;
    std::cin >> x;
    fen.add(i, x);
  }
  while (Q--) {
    i32 t, a, b;
    std::cin >> t >> a >> b;
    if (t == 0) {
      fen.add(a, b);
    }
    else {
      std::cout << fen.fold(a, b) << '\n';
    }
  }
  return 0;
}
