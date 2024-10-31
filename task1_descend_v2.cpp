#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
 #include <chrono>
 
int main() {
    auto start = std::chrono::high_resolution_clock::now();
    // Параметры сетки
    int Nx = 40; 
    int Ny = 40; 
    double x_min = -1.0;
    double x_max = 1.0;
    double y_min = -0.5;
    double y_max = 0.5;
    double dx = (x_max - x_min) / (Nx - 1);
    double dy = (y_max - y_min) / (Ny - 1);
 
    std::vector<std::vector<int>> index_map(Nx, std::vector<int>(Ny, -1));
    std::vector<int> idx_i, idx_j;
    std::vector<double> x_coords, y_coords;
 
    int N = 0; // число внутренних узлов
 
    for (int i = 0; i < Nx; ++i) {
        double x = x_min + i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = y_min + j * dy;
            // Проверяем, лежит ли точка внутри эллипса
            if (x * x + 4 * y * y < 1.0) {
                index_map[i][j] = N;
                idx_i.push_back(i);
                idx_j.push_back(j);
                x_coords.push_back(x);
                y_coords.push_back(y);
                ++N;
            }
        }
    }
 
    // Инициализируем параметры
    std::vector<double> u(N, 0.0); // начальное приближение
    std::vector<double> b(N, 1.0); // правая часть
    std::vector<double> r(N, 0.0); // невязка
 
    // Структура матрицы
    struct Entry {
        int row;
        int col;
        double value;
    };
    std::vector<Entry> A_entries;
 
    for (int idx = 0; idx < N; ++idx) {
        int i = idx_i[idx];
        int j = idx_j[idx];
 
        double a_center = 2.0 / (dx * dx) + 2.0 / (dy * dy);
        A_entries.push_back({idx, idx, a_center});
 
        if (i > 0) {
            int idx_left = index_map[i - 1][j];
            if (idx_left != -1) {
                double a_left = -1.0 / (dx * dx);
                A_entries.push_back({idx, idx_left, a_left});
            }
        }
        if (i < Nx - 1) {
            int idx_right = index_map[i + 1][j];
            if (idx_right != -1) {
                double a_right = -1.0 / (dx * dx);
                A_entries.push_back({idx, idx_right, a_right});
            }
        }
 
        if (j > 0) {
            int idx_down = index_map[i][j - 1];
            if (idx_down != -1) {
                double a_down = -1.0 / (dy * dy);
                A_entries.push_back({idx, idx_down, a_down});
            }
        }
        if (j < Ny - 1) {
            int idx_up = index_map[i][j + 1];
            if (idx_up != -1) {
                double a_up = -1.0 / (dy * dy);
                A_entries.push_back({idx, idx_up, a_up});
            }
        }
    }
 
    // Метод наискорейшего спуска
    int max_iter = 50000;
    double tol = 1e-5;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Вычисляем r = b - A * u
        std::fill(r.begin(), r.end(), 0.0);
        for (auto &entry : A_entries) {
            r[entry.row] += entry.value * u[entry.col];
        }
        for (int i = 0; i < N; ++i) {
            r[i] = b[i] - r[i];
        }
 
        double r_norm = 0.0;
        for (double val : r) {
            r_norm += val * val;
        }
        r_norm = std::sqrt(r_norm);
        if (r_norm < tol) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
 
        std::vector<double> Ar(N, 0.0);
        for (auto &entry : A_entries) {
            Ar[entry.row] += entry.value * r[entry.col];
        }
 
        //скалярные произведения
        double rr = 0.0;
        double rAr = 0.0;
        for (int i = 0; i < N; ++i) {
            rr += r[i] * r[i];
            rAr += r[i] * Ar[i];
        }
 
        //шаг 
        double alpha = rr / rAr;
 
        for (int i = 0; i < N; ++i) {
            u[i] += alpha * r[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now(); 
    
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Время выполнения: " << duration.count() << " мс" << std::endl;

    std::ofstream outfile("solution_v2_40.csv");
    outfile << "x,y,u\n";
    outfile << std::fixed << std::setprecision(6);
    for (int idx = 0; idx < N; ++idx) {
        outfile << x_coords[idx] << "," << y_coords[idx] << "," << u[idx] << "\n";
    }
    outfile.close();
 
    return 0;
}
