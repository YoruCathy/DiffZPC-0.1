# DiffZPC-0.1
```
I'm currently doing this project just for fun :) Updates will be made on an irregular basis.
```
This project adds auto differentiation to ZPC(zenus parallel computing library). ZPC is a *modern fancy C++* library for parallel computing on multiple platforms. It delivers great parallel computing efficiency for physics-based simulations within a shared-memory heterogeneous architecture through a unified programming interface on multiple compute backends. 
![zpc_structure](https://user-images.githubusercontent.com/16759982/176507009-1e83cbf0-9a75-4ce2-a224-67143a206023.png)

# Requirements
Please refer to [zpc](https://github.com/zenustech/zpc) for detailed install requirements. Here we provide a brief cmake guide:
```
mkdir build && cd build
cmake ..
make -j8
```
This works for gcc-10.3.0, cuda-11.6.

