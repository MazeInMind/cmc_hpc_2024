#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <cassert>
 
const int N = 40;           // Размер глобальной сетки (N x N)
const double hx = 2.0 / N;  // Шаг по x (от -1 до 1)
const double hy = 1.0 / N;  // Шаг по y (от -0.5 до 0.5)
const double tolerance = 1e-5; // Критерий сходимости
 
double f(double x, double y) {
    return 1.0;
}
 
bool insideEllipse(double x, double y) {
    return (x * x + 4 * y * y) < 1.0;
}
 
inline int idx(int i, int j, int local_Ny) {
    return i * (local_Ny + 2) + j;
}
 
// Итерационный шаг метода Якоби для решения уравнения Пуассона
double solvePoissonLocal(std::vector<double> &u_new,
                         const std::vector<double> &u_old,
                         int local_Nx, int local_Ny,
                         double x_start, double y_start,
                         bool &local_converged) {
    double local_max_diff = 0.0;
    local_converged = true;
 
    #pragma omp parallel
    {
        double thread_max_diff = 0.0;
        bool thread_converged = true;
 
        #pragma omp for nowait
        for (int i = 1; i <= local_Nx; ++i) {
            for (int j = 1; j <= local_Ny; ++j) {
                double x = x_start + (j - 1) * hx;
                double y = y_start + (i - 1) * hy;
                int current = idx(i, j, local_Ny);
 
                if (insideEllipse(x, y)) {
                    int up = idx(i - 1, j, local_Ny);
                    int down = idx(i + 1, j, local_Ny);
                    int left = idx(i, j - 1, local_Ny);
                    int right = idx(i, j + 1, local_Ny);
 
                    double val = 0.25 * (u_old[up] + u_old[down] +
                                         u_old[left] + u_old[right] +
                                         hx * hy * f(x, y));
                    double diff = std::fabs(val - u_old[current]);
                    u_new[current] = val;
 
                    if (diff > tolerance) {
                        thread_converged = false;
                    }
                    if (diff > thread_max_diff) {
                        thread_max_diff = diff;
                    }
                } else {
                    // На границе эллипса: граничные условия Дирихле (u=0)
                    u_new[current] = 0.0;
                }
            }
        }
 
        #pragma omp critical
        {
            if (thread_max_diff > local_max_diff) {
                local_max_diff = thread_max_diff;
            }
            if (!thread_converged) {
                local_converged = false;
            }
        }
    }
 
    return local_max_diff;
}
 
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
 
    double start = MPI_Wtime();
 
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    int Px = 1, Py = 1;
    if (size == 2) { Px = 2; Py = 1; }
    if (size == 4) { Px = 2; Py = 2; }
 
    int dims[2] = {Py, Px};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
 
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int proc_row = coords[0];
    int proc_col = coords[1];
 
    if (N % Px != 0 || N % Py != 0) {
        if (rank == 0) {
            std::cerr << "Размер сетки N должен делиться на Px и Py без остатка." << std::endl;
        }
        MPI_Finalize();
        return -1;
    }
 
    int local_Nx = N / Py;
    int local_Ny = N / Px;
 
    double x_start = -1.0 + proc_col * local_Ny * hx;
    double y_start = -0.5 + proc_row * local_Nx * hy;
 
    int north, south, east, west;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);
 
    std::vector<double> u_old((local_Nx + 2) * (local_Ny + 2), 0.0);
    std::vector<double> u_new((local_Nx + 2) * (local_Ny + 2), 0.0);
 
    MPI_Datatype row_type, column_type;
    MPI_Type_contiguous(local_Ny, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);
    MPI_Type_vector(local_Nx, 1, local_Ny + 2, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);
 
    bool converged = false;
    int iteration = 0;
 
    while (!converged) {
        iteration++;
 
        MPI_Request requests[8];
        int req_count = 0;
 
        // По оси y
        if (north != MPI_PROC_NULL) {
            MPI_Isend(&u_old[idx(1, 1, local_Ny)], 1, row_type, north, 0, cart_comm, &requests[req_count++]);
            MPI_Irecv(&u_old[idx(0, 1, local_Ny)], 1, row_type, north, 1, cart_comm, &requests[req_count++]);
        } else {
            for (int j = 1; j <= local_Ny; ++j) {
                u_old[idx(0, j, local_Ny)] = 0.0;
            }
        }
 
        if (south != MPI_PROC_NULL) {
            MPI_Isend(&u_old[idx(local_Nx, 1, local_Ny)], 1, row_type, south, 1, cart_comm, &requests[req_count++]);
            MPI_Irecv(&u_old[idx(local_Nx + 1, 1, local_Ny)], 1, row_type, south, 0, cart_comm, &requests[req_count++]);
        } else {
            for (int j = 1; j <= local_Ny; ++j) {
                u_old[idx(local_Nx + 1, j, local_Ny)] = 0.0;
            }
        }
 
        // По оси x
        if (west != MPI_PROC_NULL) {
            MPI_Isend(&u_old[idx(1, 1, local_Ny)], 1, column_type, west, 2, cart_comm, &requests[req_count++]);
            MPI_Irecv(&u_old[idx(1, 0, local_Ny)], 1, column_type, west, 3, cart_comm, &requests[req_count++]);
        } else {
            for (int i = 1; i <= local_Nx; ++i) {
                u_old[idx(i, 0, local_Ny)] = 0.0;
            }
        }
 
        if (east != MPI_PROC_NULL) {
            MPI_Isend(&u_old[idx(1, local_Ny, local_Ny)], 1, column_type, east, 3, cart_comm, &requests[req_count++]);
            MPI_Irecv(&u_old[idx(1, local_Ny + 1, local_Ny)], 1, column_type, east, 2, cart_comm, &requests[req_count++]);
        } else {
            for (int i = 1; i <= local_Nx; ++i) {
                u_old[idx(i, local_Ny + 1, local_Ny)] = 0.0;
            }
        }
 
        // Ожидание завершения обмена
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
 
 
        bool local_converged = false;
        double local_max_diff = solvePoissonLocal(u_new, u_old, local_Nx, local_Ny, x_start, y_start, local_converged);
 
        double global_max_diff = 0.0;
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
 
        converged = (global_max_diff < tolerance);
 
        u_old.swap(u_new);
 
        if (rank == 0) {
            std::cout << "Итерация: " << iteration << ", глобальная погрешность: " << global_max_diff << std::endl;
        }
 
        if (iteration > 10000) {
            if (rank == 0) std::cerr << "Слишком много итераций!" << std::endl;
            break;
        }
    }
 
    // Сбор решения на процессе 0
    std::vector<double> local_u(local_Nx * local_Ny, 0.0);
    for (int i = 1; i <= local_Nx; ++i) {
        for (int j = 1; j <= local_Ny; ++j) {
            local_u[(i - 1) * local_Ny + (j - 1)] = u_old[idx(i, j, local_Ny)];
        }
    }
 
    int local_size = local_Nx * local_Ny;
    std::vector<int> recvcounts(size, 0);
    std::vector<int> displs(size, 0);
 
    MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, cart_comm);
 
    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }
 
    std::vector<double> global_u;
    if (rank == 0) {
        global_u.resize(N * N, 0.0);
    }
 
    MPI_Gatherv(local_u.data(), local_size, MPI_DOUBLE,
                global_u.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, cart_comm);
 
    if (rank == 0) {
        std::vector<double> full_grid(N * N, 0.0);
        for (int proc = 0; proc < size; ++proc) {
            int c[2];
            MPI_Cart_coords(cart_comm, proc, 2, c);
            int pr = c[0];
            int pc = c[1];
 
            int offset = displs[proc];
            for (int i = 0; i < local_Nx; ++i) {
                for (int j = 0; j < local_Ny; ++j) {
                    int global_i = pr * local_Nx + i;
                    int global_j = pc * local_Ny + j;
                    int global_idx = global_i * N + global_j;
                    int local_idx = i * local_Ny + j;
                    full_grid[global_idx] = global_u[offset + local_idx];
                }
            }
        }
 
        std::ofstream file("solution.csv");
        file << std::fixed << std::setprecision(5);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                file << full_grid[i * N + j];
                if (j != N - 1) {
                    file << ",";
                } else {
                    file << "\n";
                }
            }
        }
        file.close();
        std::cout << "Решение сохранено в файл solution.csv" << std::endl;
    }
 
    MPI_Type_free(&row_type);
    MPI_Type_free(&column_type);
 
    double end = MPI_Wtime();
    double fin_time = end - start;
    if (rank == 0) {
        std::cout << "Время вычислений: " << fin_time << " секунд" << std::endl;
    }
 
    MPI_Finalize();
    return 0;
}
