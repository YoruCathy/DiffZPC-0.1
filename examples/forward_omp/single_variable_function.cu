#include <cuda_runtime.h>
#include <iostream>
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/cuda/Cuda.h"

#include "zensim/container/Vector.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "zensim/autodiff/forward/dual.hpp"
//using namespace autodiff;

constexpr autodiff::dual f(autodiff::dual x)
{
autodiff::dual tmp = 1 + x + x*x;
static_assert(autodiff::detail::isExpr<RM_CVREF_T(tmp)>, "this is indeed an expr");
return tmp;
}

int main() {
    using namespace zs;
    fmt::print("\n\n==========example-forward-single-variable-function===========\n\n");
    auto ompPol=omp_exec().profile(true);
    Vector<autodiff::dual> val{1, memsrc_e::host};
    Vector<autodiff::dual> val1{1, memsrc_e::host};
    for(int i=0; i!=val.size();++i){
        val[i]=10.0;
    }
    for(int i=0; i!=val.size();++i){
        std::cout<<"val[i]"<<val[i]<<std::endl;
    }

    ompPol(zip(val,val1), [](auto &val, auto &val1) mutable {
        printf("val: %f\n", val.val);
        auto tmp = derivative(f, wrt(val), at(val));;
        printf("val.grad: (%f)\n", val.grad);
        printf("val.grad2: (%f)\n", tmp);
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