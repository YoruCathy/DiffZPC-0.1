#include <cuda_runtime.h>
#include <iostream>
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/cuda/Cuda.h"

#include "zensim/container/Vector.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"

#include "zensim/autodiff/forward/dual.hpp"
//using namespace autodiff;

constexpr autodiff::dual f(autodiff::dual x)
{
//  autodiff::dual tmp = 1 + x + x*x;
//autodiff::dual tmp = 1;
//tmp += std::move(x);
//autodiff::dual tmp = autodiff::detail::AddExpr<autodiff::detail::PreventExprRef<int>, autodiff::detail::PreventExprRef<autodiff::dual>>{ 1, x };
//autodiff::dual tmp = autodiff::detail::BinaryExpr<autodiff::detail::AddOp, int, autodiff::dual>{1, x};
#if 0
    autodiff::dual tmp{};// = autodiff::detail::BinaryExpr<autodiff::detail::AddOp, int, int>{1, 2};
    assign(tmp, autodiff::detail::BinaryExpr<autodiff::detail::AddOp, int, autodiff::dual>{1, x});
#elif 1
  auto tmp = autodiff::dual::def(x + 1 + x * x);
#else
  autodiff::dual tmp{};
  assign(tmp, autodiff::detail::BinaryExpr<autodiff::detail::AddOp, int, int>{1, 2});
#endif
  //  static_assert(autodiff::detail::isExpr<RM_CVREF_T(tmp)>, "this is indeed an expr");
  return tmp;
}

int main() {
  using namespace zs;
  fmt::print("\n\n==========example-forward-single-variable-function===========\n\n");
  auto cudaPol= cuda_exec().profile(true);
  Vector<autodiff::dual> val{1, memsrc_e::device, 0};

  cudaPol(enumerate(val), [] __device__(auto id, auto &v) mutable {
    v = 10;
    printf("v: %f", (float)v);
  });
  cudaPol(Collapse{val.size()}, [val = proxy<execspace_e::cuda>(val)] __device__(auto id) mutable {
    autodiff::dual v_ = val[id];
    printf("v_: %f", (float)v_);
    //    autodiff::dual* v = &v_;
    double dudx = derivative(f, wrt(v_), at(v_));
    printf("val.grad: (%f)\n", dudx);
    auto u = f(v_);
    printf("u: (%f)\n", (float)u);
  });
  /* -----old example ------*/
  //  autodiff::dual x = 2.0;                                 // the input variable x
  //  autodiff::dual u = f(x);                                // the output variable u
  //
  //  double dudx = derivative(f, wrt(x), at(x));   // evaluate the derivative du/dx
  //
  //  std::cout << "u = " << u << std::endl;        // print the evaluated output u
  //  std::cout << "du/dx = " << dudx << std::endl; // print the evaluated derivative du/dx

  return 0;
}