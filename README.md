# PBM_Schemes
This repository contains scripts that can be used to explore the numerical solution of a variety of population balance models in 1D (in Matlab) and 2D (in Python). In the 1D case, three numerical schemes are considered: the Upwind, Lax-Wendroff and Leapfrog schemes. In the 2D case, variations of the upwind scheme and scripts using various declarative solvers in Python are considered. In many instances, if the upwind scheme is applied together with variable transformations, specially constructed meshes, and operator splitting (in the case of 2D PBMs), it is possible to solve the PBM to machine precision very efficiently. 

## Citing

If you found the material in this repository useful for your work, please consider citing the following articles:

1. Pavan K. Inguva, Kaylee C. Schickel, and Richard D. Braatz (Apr. 2022). “Efficient numerical schemes for population balance models”. In: Computers & Chemical Engineering, p. 107808 (https://doi.org/10.1016/j.compchemeng.2022.107808)
2. Pavan Inguva and Richard D. Braatz (Jun. 2022). "Efficient Numerical Schemes for Multidimensional Population Balance Models". 	arXiv:2206.12404 (https://arxiv.org/abs/2206.12404) 

If you require some assistance learning how to use FiPy to solve PBMs and other PDEs, consider exploring the repo https://github.com/pavaninguva/pde-Solver-Course and reviewing the following paper:

Pavan Inguva, Vijesh J. Bhute, Thomas N.H. Cheng, and Pierre J. Walker (July 2021). “Introducing students to research codes: A short course on solving partial differential equations in Python”. In: Education for Chemical Engineers 36, pp. 1–11 (https://doi.org/10.1016/j.ece.2021.01.011)

## License

The content in this repository is provided under a CC-BY-ND 4.0 license. Please refer to the `LICENSE` file for more information.


