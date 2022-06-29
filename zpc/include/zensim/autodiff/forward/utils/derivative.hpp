//
// Created by cathy on 6/7/22.
//

#ifndef ZPC_DERIVATIVE_HPP
#define ZPC_DERIVATIVE_HPP


// C++ includes
#include <cassert>
#include <cstddef>
#include "zensim/types/Tuple.h"

// autodiff includes
#include "zensim/autodiff/common/meta.hpp"
#include "zensim/autodiff/common/vectortraits.hpp"
#include "zensim/TypeAlias.hpp"

#pragma once

namespace autodiff {
  namespace detail {
    template<typename... Args>
    struct At
    {
      zs::tuple<Args...> args;
//      std::tuple<Args...> args;
    };

    template<typename... Args>
    struct Wrt
    {
      zs::tuple<Args...> args;
//        std::tuple<Args...> args;
    };

    template<typename... Args>
    struct Along
    {
      zs::tuple<Args...> args;
//        std::tuple<Args...> args;
    };

    /// The keyword used to denote the variables *with respect to* the derivative is calculated.
    template<typename... Args>
    constexpr auto wrt(Args&&... args)
    {
      return Wrt<Args&&...>{ zs::forward_as_tuple(FWD(args)...) };
    }

    /// The keyword used to denote the variables *at* which the derivatives are calculated.
    template<typename... Args>
    constexpr auto at(Args&&... args)
    {
      return At<Args&&...>{ zs::forward_as_tuple(FWD(args)...) };
      //return At<Args&&...>{ zs::tuple<Args&&...>(FWD(args)...) };
    }

    /// The keyword used to denote the direction vector *along* which the derivatives are calculated.
    template<typename... Args>
    constexpr auto along(Args&&... args)
    {
      return Along<Args&&...>{ zs::forward_as_tuple(FWD(args)...) };
    }

    /// Seed each dual number in the **wrt** list using its position as the derivative order to be seeded.
    /// Using `seed(wrt(x, y, z), 1)` will set the 1st order derivative of `x`, the
    /// 2nd order derivative of `y`, and the 3rd order derivative of `z` to 1. If
    /// these dual numbers have order greater than 3, then the last dual number will
    /// be used for the remaining higher-order derivatives. For example, if these
    /// numbers are 5th order, than the 4th and 5th order derivatives of `z` will be
    /// set to 1 as well. In this example, `wrt(x, y, z)` is equivalent to `wrt(x,
    /// y, z, z, z)`. This automatic seeding permits derivatives `fx`, `fxy`,
    /// `fxyz`, `fxyzz`, and `fxyzzz` to be computed in a more convenient way.
    template<typename Var, typename... Vars, typename T>
//    constexpr auto seed(const Wrt<Var&, Vars&...>& wrt, T&& seedval)
//    tatic assertion failed with "It is not possible to compute higher-order derivatives with order greater than that of the autodiff number (e.g., using wrt(x, x, y, z) will fail if the autodiff numbers in use have order below 4)."
    constexpr auto seed(const Wrt<Var&, Vars&...>& wrt, T&& seedval)
    {
      constexpr auto N = Order<Var>;
      constexpr auto size = 1 + sizeof...(Vars);

      static_assert(size <= N, "It is not possible to compute higher-order derivatives with order greater than that of the autodiff number (e.g., using wrt(x, x, y, z) will fail if the autodiff numbers in use have order below 4).");
      For<N>([&wrt, &seedval](auto i) constexpr {
        constexpr auto i_ = std::remove_cv_t<std::remove_const_t<decltype(i)>>::index;
        if constexpr (i_ < size)
          seed<i_ + 1>(zs::get<i_>(wrt.args), seedval);
        else
          seed<i_ + 1>(zs::get<size - 1>(wrt.args), seedval); // use the last variable in the wrt list as the variable for which the remaining higher-order derivatives are calculated (e.g., derivatives(f, wrt(x), at(x)) will produce [f0, fx, fxx, fxxx, fxxxx] when x is a 4th order dual number).
      });
    }

    template<typename... Vars>
    constexpr auto seed(const Wrt<Vars...>& wrt)
    {
      seed(wrt, 1.0);
    }

    template<typename... Vars>
    constexpr auto unseed(const Wrt<Vars...>& wrt)
    {
      seed(wrt, 0.0);
    }

    template<typename... Args, typename... Vecs>
    constexpr auto seed(const At<Args...>& at, const Along<Vecs...>& along)
    {
      static_assert(sizeof...(Args) == sizeof...(Vecs));

      ForEach(at.args, along.args, [&](auto& arg, auto&& dir) constexpr {
            if constexpr (isVector<decltype(arg)>) {
              static_assert(isVector<decltype(dir)>);
              assert(arg.size() == dir.size());
              for(auto i = 0; i < dir.size(); ++i)
                seed<1>(arg[i], dir[i]);
            }
            else seed<1>(arg, dir);
          });
    }

    template<typename... Args>
    constexpr auto unseed(const At<Args...>& at)
    {
      ForEach(at.args, [&](auto& arg) constexpr {
            if constexpr (isVector<decltype(arg)>) {
              for(auto i = 0; i < arg.size(); ++i)
                seed<1>(arg[i], 0.0);
            }
            else seed<1>(arg, 0.0);
          });
    }

    template<size_t order = 1, typename T, EnableIf<Order<T>>...>
    constexpr auto seed(T& x)
    {
      seed<order>(x, 1.0);
    }

    template<size_t order = 1, typename T, EnableIf<Order<T>>...>
    constexpr auto unseed(T& x)
    {
      seed<order>(x, 0.0);
    }

    template<typename Fun, typename... Args, typename... Vars>
    constexpr auto eval(const Fun& f, const At<Args...>& at, const Wrt<Vars...>& wrt)
    {
      seed(wrt);
      auto u = zs::apply(f, at.args);
      unseed(wrt);
      return u;
    }

    template<typename Fun, typename... Args, typename... Vecs>
    constexpr auto eval(const Fun& f, const At<Args...>& at, const Along<Vecs...>& along)
    {
      seed(at, along);
      auto u = zs::apply(f, at.args);
      unseed(at);
      return u;
    }

    /// Extract the derivative of given order from a vector of dual/real numbers.
    template<size_t order = 1, typename Vec, EnableIf<isVector<Vec>>...>
    constexpr auto derivative(const Vec& u)
    {
      printf("Inside derivative2: u len = %d, %f\n", u.size(), u[0]);
      size_t len = u.size(); // the length of the vector containing dual/real numbers
      using NumType = decltype(u[0]); // get the type of the dual/real number
      using T = NumericType<NumType>; // get the numeric/floating point type of the dual/real number
      using Res = VectorReplaceValueType<Vec, T>; // get the type of the vector containing numeric values instead of dual/real numbers (e.g., vector<real> becomes vector<double>, VectorXdual becomes VectorXd, etc.)
      Res res(len); // create an array to store the derivatives stored inside the dual/real number
      for(auto i = 0; i < len; ++i)
        res[i] = derivative<order>(u[i]); // get the derivative of given order from i-th dual/real number
      return res;
    }

    /// Alias method to `derivative<order>(x)` where `x` is either a dual/real number or vector/array of such numbers.
    template<size_t order = 1, typename T>
    constexpr auto grad(const T& x)
    {
      return derivative<order>(x);
    }

    /// Unpack the derivatives from the result of an @ref eval call into an array.
    template<typename Result>
    constexpr auto derivatives(const Result& result)
    {
      if constexpr (isVector<Result>) // check if the argument is a vector container of dual/real numbers
      {
        size_t len = result.size(); // the length of the vector containing dual/real numbers
        using NumType = decltype(result[0]); // get the type of the dual/real number
        using T = NumericType<NumType>; // get the numeric/floating point type of the dual/real number
        using Vec = VectorReplaceValueType<Result, T>; // get the type of the vector containing numeric values instead of dual/real numbers (e.g., vector<real> becomes vector<double>, VectorXdual becomes VectorXd, etc.)
        constexpr auto N = Order<NumType>; // the order of the dual/real number
        std::array<Vec, N + 1> values; // create an array to store the derivatives stored inside the dual/real number
        For<N + 1>([&](auto i) constexpr {
          values[i].resize(len);
          for(auto j = 0; j < len; ++j)
            values[i][j] = derivative<i>(result[j]); // get the ith derivative of the jth dual/real number
        });
        return values;
      }
      else // result is then just a dual/real number
      {
        using T = NumericType<Result>; // get the numeric/floating point type of the dual/real result number
        constexpr auto N = Order<Result>; // the order of the dual/real result number
        std::array<T, N + 1> values; // create an array to store the derivatives stored inside the dual/real number
        For<N + 1>([&](auto i) constexpr {
          values[i] = derivative<i>(result);
        });
        return values;
      }
    }

    template<typename Fun, typename... Vars, typename... Args>
    constexpr auto derivatives(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at)
    {
      return derivatives(eval(f, at, wrt));
    }

    template<size_t order=1, typename Fun, typename... Vars, typename... Args, typename Result>
    constexpr auto derivative(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, Result& u)
    {
      u = derivatives(f, wrt, at);
      return derivative<order>(u);
    }

    template<size_t order=1, typename Fun, typename... Vars, typename... Args>
    constexpr auto derivative(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at)
    {
      auto u = eval(f, at, wrt);
      printf("In derivative: u = %f\n", u);
      return derivative<order>(u);
    }

    template<typename Fun, typename... Vecs, typename... Args>
    constexpr auto derivatives(const Fun& f, const Along<Vecs...>& along, const At<Args...>& at)
    {
      return derivatives(eval(f, at, along));
    }

  } // namespace detail

  using detail::derivatives;
  using detail::derivative;
  using detail::grad;
  using detail::along;
  using detail::wrt;
  using detail::at;
  using detail::seed;
  using detail::unseed;

  using detail::Along;
  using detail::At;
  using detail::Wrt;

} // namespace autodiff


#endif  // ZPC_DERIVATIVE_HPP
