//
// Created by cathy on 6/7/22.
//

#ifndef ZPC_META_HPP
#define ZPC_META_HPP

#pragma once

// C++ includes
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace autodiff {
  namespace detail {

    template<bool value>
    using EnableIf = typename std::enable_if<value>::type;

    template<typename T>
    using PlainType = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

    template<bool Cond, typename WhenTrue, typename WhenFalse>
    using ConditionalType = typename std::conditional<Cond, WhenTrue, WhenFalse>::type;

    template<typename A, typename B>
    using CommonType = typename std::common_type<A, B>::type;

    template<typename Fun, typename... Args>
    using ReturnType = std::invoke_result_t<Fun, Args...>;

    template<typename T, typename U>
    constexpr bool isConvertible = std::is_convertible<PlainType<T>, U>::value;

    template<typename A, typename B>
    constexpr bool isSame = std::is_same_v<A, B>;

    template<typename Tuple>
    constexpr auto TupleSize = std::tuple_size_v<std::decay_t<Tuple>>;

    template<typename Tuple>
    constexpr auto TupleHead(Tuple&& tuple)
    {
      return std::get<0>(std::forward<Tuple>(tuple));
    }

    template<typename Tuple>
    constexpr auto TupleTail(Tuple&& tuple)
    {
      auto g = [&](auto&&, auto&&... args) constexpr {
        return std::forward_as_tuple(args...);
      };
      return std::apply(g, std::forward<Tuple>(tuple));
    }

    template<size_t i>
    struct Index
    {
      constexpr static size_t index = i;
      constexpr operator size_t() const { return index; }
      constexpr operator size_t() { return index; }
    };

    template<size_t i, size_t ibegin, size_t iend, typename Function>
    constexpr auto AuxFor(Function&& f)
    {
      if constexpr (i < iend) {
        f(Index<i>{});
        AuxFor<i + 1, ibegin, iend>(std::forward<Function>(f));
      }
    }

    template<size_t ibegin, size_t iend, typename Function>
    constexpr auto For(Function&& f)
    {
      AuxFor<ibegin, ibegin, iend>(std::forward<Function>(f));
    }

    template<size_t iend, typename Function>
    constexpr auto For(Function&& f)
    {
      For<0, iend>(std::forward<Function>(f));
    }

    template<size_t i, size_t ibegin, size_t iend, typename Function>
    constexpr auto AuxReverseFor(Function&& f)
    {
      if constexpr (i < iend)
      {
        AuxReverseFor<i + 1, ibegin, iend>(std::forward<Function>(f));
        f(Index<i>{});
      }
    }

    template<size_t ibegin, size_t iend, typename Function>
    constexpr auto ReverseFor(Function&& f)
    {
      AuxReverseFor<ibegin, ibegin, iend>(std::forward<Function>(f));
    }

    template<size_t iend, typename Function>
    constexpr auto ReverseFor(Function&& f)
    {
      ReverseFor<0, iend>(std::forward<Function>(f));
    }

    template<typename Tuple, typename Function>
    constexpr auto ForEach(Tuple&& tuple, Function&& f)
    {
      constexpr auto N = TupleSize<Tuple>;
      For<N>([&](auto i) constexpr {
        f(std::get<i>(tuple));
      });
      //------------------------------------------------------------
      // ALTERNATIVE IMPLEMENTATION POSSIBLY USEFUL TO KEEP IT HERE
      // auto g = [&](auto&&... args) constexpr {
      //     ( f(std::forward<decltype(args)>(args)), ...);
      // };
      // std::apply(g, std::forward<Tuple>(tuple));
      //------------------------------------------------------------
    }

    template<typename Tuple1, typename Tuple2, typename Function>
    constexpr auto ForEach(Tuple1&& tuple1, Tuple2&& tuple2, Function&& f)
    {
      constexpr auto N1 = TupleSize<Tuple1>;
      constexpr auto N2 = TupleSize<Tuple2>;
      static_assert(N1 == N2);
      For<N1>([&](auto i) constexpr {
        f(std::get<i>(tuple1), std::get<i>(tuple2));
      });
    }

    template<size_t ibegin, size_t iend, typename Function>
    constexpr auto Sum(Function&& f)
    {
      using ResultType = std::invoke_result_t<Function, Index<ibegin>>;
      ResultType res = {};
      For<ibegin, iend>([&](auto i) constexpr {
        res += f(Index<i>{});
      });
      return res;
    }

    template<typename Tuple, typename Function>
    constexpr auto Reduce(Tuple&& tuple, Function&& f)
    {
      using ResultType = std::invoke_result_t<Function, decltype(std::get<0>(tuple))>;
      ResultType res = {};
      ForEach(tuple, [&](auto&& item) constexpr {
            res += f(item);
          });
      return res;
    }

  } // namespace detail
} // namespace autodiff


#endif  // ZPC_META_HPP
