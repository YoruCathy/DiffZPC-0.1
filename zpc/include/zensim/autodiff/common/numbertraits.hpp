#ifndef ZPC_NUMBERTRAITS_HPP
#define ZPC_NUMBERTRAITS_HPP


#pragma once

// C++ includes
#include "zensim/autodiff/common/meta.hpp"

namespace autodiff {
  namespace detail {

    /// A trait class used to specify whether a type is arithmetic.
    template<typename T>
    struct ArithmeticTraits
    {
      static constexpr bool isArithmetic = std::is_arithmetic_v<T>;
    };

    /// A compile-time constant that indicates whether a type is arithmetic.
    template<typename T>
    constexpr bool isArithmetic = ArithmeticTraits<PlainType<T>>::isArithmetic;

    /// An auxiliary template type to indicate NumberTraits has not been defined for a type.
    template<typename T>
    struct NumericTypeInfoNotDefinedFor { using type = T; };

    /// A trait class used to specify whether a type is an autodiff number.
    template<typename T>
    struct NumberTraits
    {
      /// The underlying floating point type of the autodiff number type.
      using NumericType = std::conditional_t<isArithmetic<T>, T, NumericTypeInfoNotDefinedFor<T>>;

      /// The order of the autodiff number type.
      static constexpr auto Order = 0;
    };

    /// A template alias to get the underlying floating point type of an autodiff number.
    template<typename T>
    using NumericType = typename NumberTraits<PlainType<T>>::NumericType;

    /// A compile-time constant with the order of an autodiff number.
    template<typename T>
    constexpr auto Order = NumberTraits<PlainType<T>>::Order;

  } // namespace detail
} // namespace autodiff


#endif  // ZPC_NUMBERTRAITS_HPP
