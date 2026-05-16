#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <fstream>
#include <string>
#include "cuda_worker.h"

using namespace std;

#define MATRIX_SIZE 50

int main(int argc, char** argv) {
    int provided;
    // We use MPI_Init_thread instead of MPI_Init because we are mixing MPI with OpenMP threads.
    // MPI_THREAD_MULTIPLE ensures that multiple threads can safely make MPI calls at the same time.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Gets the ID of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Gets the total number of processes

    // This program is strictly designed to run with exactly 11 processes (Master + 10 Workers)
    if (size != 11) {
        if (rank == 0) cout << "Please run with exactly 11 processes (-n 11)." << endl;
        MPI_Finalize();
        return 0;
    }

    // ==========================================
    // RANK 0: THE MASTER PROCESS
    // ==========================================
    if (rank == 0) {
        cout << "Master starting..." << endl;

        // --- Task 1: Send a number to Rank 1 for Factorial calculation ---
        int my_int = 5;
        MPI_Send(&my_int, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // --- Task 2: Send a string to Rank 2 for Vowel counting ---
        char my_str[100] = "Parallel Programming is fun";
        MPI_Send(my_str, 100, MPI_CHAR, 2, 0, MPI_COMM_WORLD);

        // --- Task 3: Create a file and send the filename to Rank 3 for splitting ---
        char filename[100] = "data.txt";
        ofstream f(filename);
        for (int i = 0; i < 10; i++) f << "Line " << i << "\n";
        f.close();
        MPI_Send(filename, 100, MPI_CHAR, 3, 0, MPI_COMM_WORLD);

        // --- Task 4 & 10: Initialize Matrices for Math and CUDA tasks ---
        // 'static' is used because these are large arrays and it keeps them off the limited stack memory
        static int mat1[MATRIX_SIZE][MATRIX_SIZE];
        static int mat2[MATRIX_SIZE][MATRIX_SIZE];
        static int cuda_mat[MATRIX_SIZE][MATRIX_SIZE];

        // Fill matrices with dummy data using OpenMP to speed up initialization
#pragma omp parallel for
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < 50; j++) {
                mat1[i][j] = 1;
                mat2[i][j] = 2;
                cuda_mat[i][j] = 5;
            }
        }

        // Send matrices to Rank 4 (The Sub-Master for Matrix Math)
        MPI_Send(mat1, 50 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD);
        MPI_Send(mat2, 50 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD);

        // Send the CUDA matrix directly to Rank 10
        MPI_Send(cuda_mat, 50 * 50, MPI_INT, 10, 0, MPI_COMM_WORLD);

        // --- Gathering Results ---

        // Get Factorial from Rank 1
        int factorial;
        MPI_Recv(&factorial, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 1 (Factorial): " << factorial << endl;

        // Get Vowel Count from Rank 2
        int vowels;
        MPI_Recv(&vowels, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 2 (Vowels): " << vowels << endl;

        // Get File Status from Rank 3
        int status;
        MPI_Recv(&status, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 3 (Files): Done." << endl;

        // Get Processed Matrix from Rank 4 and print it
        static int mat_result[50][50];
        MPI_Recv(mat_result, 50 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 4 (Matrix complete). The result: " << endl;
        for (auto& i : mat_result)
        {
            for (auto& j : i) {
                cout << j << " ";
            }
            cout << endl;
        }
        cout << endl;

        // Get CUDA Processed Matrix from Rank 10 and print it
        MPI_Recv(cuda_mat, 50 * 50, MPI_INT, 10, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 10 (CUDA Bonus). Matrix multiplied by 10: " << endl;
        for (auto& i : cuda_mat)
        {
            for (auto& j : i) {
                cout << j << " ";
            }
            cout << endl;
        }
        cout << endl;

    }
    // ==========================================
    // RANK 1: FACTORIAL WORKER
    // ==========================================
    else if (rank == 1) {
        int data;
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int fact = 1;
        // Use OpenMP reduction to safely multiply the factorial in parallel
#pragma omp parallel for reduction(*:fact)
        for (int i = 1; i <= data; i++) fact *= i;
        MPI_Send(&fact, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    // ==========================================
    // RANK 2: VOWEL COUNTER WORKER
    // ==========================================
    else if (rank == 2) {
        char str[100];
        MPI_Recv(str, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int vowels = 0;
        string s(str);
        // Use OpenMP reduction to safely count vowels in parallel without race conditions
#pragma omp parallel for reduction(+:vowels)
        for (int i = 0; i < s.length(); i++) {
            char c = tolower(s[i]);
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') vowels++;
        }
        MPI_Send(&vowels, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    // ==========================================
    // RANK 3: FILE SPLITTER WORKER
    // ==========================================
    else if (rank == 3) {
        char filename[100];
        MPI_Recv(filename, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        string lines[100];
        int count = 0;

        // Read the file sequentially into memory
        ifstream in(filename);
        while (getline(in, lines[count]) && count < 100) { count++; }
        in.close();

        // Use OpenMP sections to assign one thread to write even lines, 
        // and a completely different thread to write odd lines simultaneously.
#pragma omp parallel sections
        {
#pragma omp section
            { ofstream even("even.txt"); for (int i = 0; i < count; i += 2) even << lines[i] << "\n"; }

#pragma omp section
            { ofstream odd("odd.txt"); for (int i = 1; i < count; i += 2) odd << lines[i] << "\n"; }
        }
        int done = 1;
        MPI_Send(&done, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    // ==========================================
    // RANK 4: THE MATRIX SUB-MASTER
    // ==========================================
    else if (rank == 4) {
        static int mat1[50][50], mat2[50][50], result[50][50];
        MPI_Recv(mat1, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(mat2, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Break the 50x50 matrix into 5 chunks of 10 rows each.
        // Send chunk 0 to Rank 5, chunk 1 to Rank 6... up to Rank 9.
        // We use OpenMP to parallelize the MPI_Send commands!
#pragma omp parallel for
        for (int dest = 5; dest <= 9; dest++) {
            int start_row = (dest - 5) * 10;
            MPI_Send(&mat1[start_row][0], 10 * 50, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&mat2[start_row][0], 10 * 50, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }

        // Receive the processed 10-row chunks back from Ranks 5-9 in parallel.
#pragma omp parallel for
        for (int source = 5; source <= 9; source++) {
            int start_row = (source - 5) * 10;
            MPI_Recv(&result[start_row][0], 10 * 50, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Send the fully reassembled 50x50 matrix back to the Master (Rank 0)
        MPI_Send(result, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    // ==========================================
    // RANKS 5 TO 9: MATRIX MATH WORKERS
    // ==========================================
    else if (rank >= 5 && rank <= 9) {
        int mat1_part[10][50], mat2_part[10][50], result_part[10][50];

        // Receive a 10x50 chunk of the matrices from Rank 4
        MPI_Recv(mat1_part, 10 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(mat2_part, 10 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Do element-wise addition on the chunk using OpenMP threads
#pragma omp parallel for
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 50; j++) result_part[i][j] = mat1_part[i][j] + mat2_part[i][j];
        }

        // Send the calculated chunk back to Rank 4
        MPI_Send(result_part, 10 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD);
    }
    // ==========================================
    // RANK 10: CUDA GPU WORKER
    // ==========================================
    else if (rank == 10) {
        static int mat[50][50];
        MPI_Recv(mat, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Call the external CUDA file/function to process this matrix on the GPU
        process_matrix_cuda(&mat[0][0]);

        // Send the GPU-processed matrix back to the Master
        MPI_Send(mat, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Clean up MPI environment
    MPI_Finalize();
    return 0;
}