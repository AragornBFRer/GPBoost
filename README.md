# Gaussian Process Boosting

This project focuses on the implementation of the idea and algorithm proposed in the paper Gaussian Process Boosting (https://arxiv.org/abs/2004.02653). 



## code/

Under this folder, we implement data_simulator.py as a simulation data generator, GPBoost.py as the implementation of the proposed GPBoost algorithm.

Run `python test.py` to get the comparison between GPBoost algorithm against LightGBM (or LSBoost) and CatBoost.

## demo/

Alternatively, to reproduce the figures used in this report, you can find the notebooks under demo/ as a convenient pathway.

`GPBoost_hybrid.ipynb`, `GPBoost_newton.ipynb`, `GPBoost_gradient.ipynb` show the result of three alternative of updating rule used in GPBoost.

`GPBoost_solution_path.ipynb` generate the solution path showed in the report.



## References

[1] Sigrist, F. (2022). Gaussian process boosting. Journal of Machine Learning Research, 23(232), 1–46.

[2] Ahlem Hajjem, F. B., & Larocque, D. (2014). Mixed-effects random forest for clustered data. Journal of Statistical Computation and Simulation, 84(6), 1313–1328.

[3] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan & R. Gar-nett (Eds.), Advances in neural information processing systems (Vol. 30). Curran Associates, Inc.
