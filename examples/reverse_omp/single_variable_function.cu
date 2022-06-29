// TODO: how to init zs::Vector, Assign value?

// C++ includes
#include <cuda_runtime.h>
#include <iostream>
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/cuda/Cuda.h"

#include "zensim/container/Vector.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
using namespace std;

// autodiff include
#include <zensim/autodiff/reverse/var.hpp>
using namespace autodiff;

// The single-variable function for which derivatives are needed
var f(var x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{
    using namespace zs;
    fmt::print("\n\n==========example-forward-single-variable-function===========\n\n");
    auto ompPol=omp_exec().profile(true);
    Vector<autodiff::var> val{1, memsrc_e::host};
    Vector<autodiff::var> val1{1, memsrc_e::host};
    for(int i=0; i!=val.size();++i){
        val[i]=2.0;
    }
    for(int i=0; i!=val1.size();++i){
        val1[i]=f(val[i]);
    }

    ompPol(zip(val,val1), [](auto &val, auto &val1) mutable {
        printf("val: %f\n", val);
        printf("val1: %f\n", val1);
        auto [tmp] = derivatives(val1, wrt(val));
        printf("uv: %f\n", tmp);
    });

//    var x = 2.0;   // the input variable x
//    var u = f(x);  // the output variable u
//
//    auto [ux] = derivatives(u, wrt(x)); // evaluate the derivative of u with respect to x
//
//    cout << "u = " << u << endl;  // print the evaluated output variable u
//    cout << "ux = " << ux << endl;  // print the evaluated derivative ux
}
