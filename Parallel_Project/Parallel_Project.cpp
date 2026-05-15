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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 11) {
        if (rank == 0) cout << "Please run with exactly 11 processes (-n 11)." << endl;
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        cout << "Master starting..." << endl;

        int my_int = 5;
        MPI_Send(&my_int, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        char my_str[100] = "Parallel Programming is fun";
        MPI_Send(my_str, 100, MPI_CHAR, 2, 0, MPI_COMM_WORLD);

        char filename[100] = "data.txt";
        ofstream f(filename);
        for (int i = 0; i < 10; i++) f << "Line " << i << "\n";
        f.close();
        MPI_Send(filename, 100, MPI_CHAR, 3, 0, MPI_COMM_WORLD);

        static int mat1[MATRIX_SIZE][MATRIX_SIZE];
        static int mat2[MATRIX_SIZE][MATRIX_SIZE];
        static int cuda_mat[MATRIX_SIZE][MATRIX_SIZE]; 

#pragma omp parallel for
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < 50; j++) {
                mat1[i][j] = 1;
                mat2[i][j] = 2;
                cuda_mat[i][j] = 5; 
            }
        }

        MPI_Send(mat1, 50 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD);
        MPI_Send(mat2, 50 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD);
        
        MPI_Send(cuda_mat, 50 * 50, MPI_INT, 10, 0, MPI_COMM_WORLD);

        int factorial;
        MPI_Recv(&factorial, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 1 (Factorial): " << factorial << endl;

        int vowels;
        MPI_Recv(&vowels, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 2 (Vowels): " << vowels << endl;

        int status;
        MPI_Recv(&status, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 3 (Files): Done." << endl;

        static int mat_result[50][50];
        MPI_Recv(mat_result, 50 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        cout << "Rank 4 (Matrix complete). The result: " << endl;
        for(auto &i:mat_result)
        {
            for(auto &j:i){
                cout<<j<<" ";
            }
            cout<<endl;
        }
        cout<<endl;

        MPI_Recv(cuda_mat, 50 * 50, MPI_INT, 10, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 10 (CUDA Bonus). Matrix multiplied by 10: " << endl;
        for(auto &i:cuda_mat)
        {
            for(auto &j:i){
                cout<<j<<" ";
            }
            cout<<endl;
        }
        cout<<endl;

    }
    else if (rank == 1) {
        int data;
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int fact = 1;
#pragma omp parallel for reduction(*:fact)
        for (int i = 1; i <= data; i++) fact *= i;
        MPI_Send(&fact, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank == 2) {
        char str[100];
        MPI_Recv(str, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int vowels = 0; string s(str);
#pragma omp parallel for reduction(+:vowels)
        for (int i = 0; i < s.length(); i++) {
            char c = tolower(s[i]);
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') vowels++;
        }
        MPI_Send(&vowels, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank == 3) {
        char filename[100];
        MPI_Recv(filename, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        string lines[100]; int count = 0;
        ifstream in(filename);
        while (getline(in, lines[count]) && count < 100) { count++; }
        in.close();
#pragma omp parallel sections
        {
#pragma omp section
            { ofstream even("even.txt"); for (int i = 0; i < count; i += 2) even << lines[i] << "\n"; }
#pragma omp section
            { ofstream odd("odd.txt"); for (int i = 1; i < count; i += 2) odd << lines[i] << "\n"; }
        }
        int done = 1; MPI_Send(&done, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank == 4) {
        static int mat1[50][50], mat2[50][50], result[50][50];
        MPI_Recv(mat1, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(mat2, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
        for (int dest = 5; dest <= 9; dest++) {
            int start_row = (dest - 5) * 10;
            MPI_Send(&mat1[start_row][0], 10 * 50, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&mat2[start_row][0], 10 * 50, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
#pragma omp parallel for
        for (int source = 5; source <= 9; source++) {
            int start_row = (source - 5) * 10;
            MPI_Recv(&result[start_row][0], 10 * 50, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Send(result, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank >= 5 && rank <= 9) {
        int mat1_part[10][50], mat2_part[10][50], result_part[10][50];
        MPI_Recv(mat1_part, 10 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(mat2_part, 10 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 50; j++) result_part[i][j] = mat1_part[i][j] + mat2_part[i][j];
        }
        MPI_Send(result_part, 10 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD);
    }
    else if (rank == 10) {
        static int mat[50][50];
        MPI_Recv(mat, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        process_matrix_cuda(&mat[0][0]);
        
        MPI_Send(mat, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}