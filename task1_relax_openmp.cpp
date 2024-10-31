#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <omp.h> 

const int N = 40; // Размер сетки
const double hx = 2.0 / N; // Шаг по x
const double hy = 1.0 / N; // Шаг по y
const double tolerance = 1e-5;

double f(double x, double y) {
    return 1.0;
}

bool insideEllipse(double x, double y) {
    return (x * x + 4 * y * y) < 1;
}

void solvePoisson(std::vector<std::vector<double>>& u) {
    bool converged = false;

    while (!converged) {

        converged = true;
        
        #pragma omp parallel for collapse(2) shared(u) reduction(&:converged)
        //Коллапсим 2 цикла в 1, все потоки имеют доступ к u, сходимость есть тогда, когда все 
        //потоки ее подтвердили
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                double x = -1.0 + i * hx;
                double y = -0.5 + j * hy;

                if (insideEllipse(x, y)) {
                    double u_new = 0.25 * (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] + hx * hy * f(x, y));
                    if (std::fabs(u_new - u[i][j]) > tolerance) {
                        converged = false;
                    }
                    u[i][j] = u_new;
                } else {
                    u[i][j] = 0.0; // Граничное условие Дирихле
                }
            }
        }
    }
}

int main() {

    std::vector<std::vector<double>> u(N, std::vector<double>(N, 0.0));

    solvePoisson(u);

    std::ofstream file("solution_100.csv");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = -1.0 + i * hx;
            double y = -0.5 + j * hy;
            file << x << "," << y << "," << u[i][j] << "\n";
        }
    }
    file.close();

    return 0;
}

