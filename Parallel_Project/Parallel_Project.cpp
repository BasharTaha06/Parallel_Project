#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <fstream>
#include <string>

using namespace std;

#define MATRIX_SIZE 50

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Force the program to run with exactly 10 processes
    if (size != 10) {
        if (rank == 0) cout << "Please run with exactly 10 processes (-n 10)." << endl;
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        // ==========================================
        // 0. MASTER PROCESS
        // ==========================================
        cout << "Master starting..." << endl;

        // 1. Send integer to Rank 1
        int my_int = 5;
        MPI_Send(&my_int, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // 2. Send string to Rank 2
        char my_str[100] = "Parallel Programming is fun";
        MPI_Send(my_str, 100, MPI_CHAR, 2, 0, MPI_COMM_WORLD);

        // 3. Create dummy file & send filename to Rank 3
        char filename[100] = "data.txt";
        ofstream f(filename);
        for (int i = 0; i < 10; i++) f << "Line " << i << "\n";
        f.close();
        MPI_Send(filename, 100, MPI_CHAR, 3, 0, MPI_COMM_WORLD);

        // 4. Send 50x50 matrices to Rank 4
        static int mat1[MATRIX_SIZE][MATRIX_SIZE];
        static int mat2[MATRIX_SIZE][MATRIX_SIZE];

        // ---> We use OpenMP in the Master here to initialize matrices <---
#pragma omp parallel for
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < 50; j++) {
                mat1[i][j] = 1;
                mat2[i][j] = 2;
            }
        }

        MPI_Send(mat1, 50 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD);
        MPI_Send(mat2, 50 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD);

        // --- Wait and Receive Results sequentially ---
        int factorial;
        MPI_Recv(&factorial, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 1 (Factorial of " << my_int << "): " << factorial << endl;

        int vowels;
        MPI_Recv(&vowels, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 2 (Vowels in string): " << vowels << endl;

        int status;
        MPI_Recv(&status, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 3 (File processed into even/odd files)." << endl;

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

    }
    else if (rank == 1) {
        // ==========================================
        // 1. INTEGER PROCESS (Factorial)
        // ==========================================
        int data;
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int fact = 1;
#pragma omp parallel for reduction(*:fact)
        for (int i = 1; i <= data; i++) {
            fact *= i;
        }

        MPI_Send(&fact, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank == 2) {
        // ==========================================
        // 2. STRING PROCESS (Count Vowels)
        // ==========================================
        char str[100];
        MPI_Recv(str, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int vowels = 0;
        string s(str);

#pragma omp parallel for reduction(+:vowels)
        for (int i = 0; i < s.length(); i++) {
            char c = tolower(s[i]);
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
                vowels++;
            }
        }

        MPI_Send(&vowels, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank == 3) {
        // ==========================================
        // 3. FILE PROCESS (Split Even/Odd lines)
        // ==========================================
        char filename[100];
        MPI_Recv(filename, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        string lines[100];
        int count = 0;
        ifstream in(filename);
        while (getline(in, lines[count]) && count < 100) { count++; }
        in.close();

#pragma omp parallel sections
        {
#pragma omp section
            {
                ofstream even("even.txt");
                for (int i = 0; i < count; i += 2) even << lines[i] << "\n";
            }
#pragma omp section
            {
                ofstream odd("odd.txt");
                for (int i = 1; i < count; i += 2) odd << lines[i] << "\n";
            }
        }

        int done = 1;
        MPI_Send(&done, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank == 4) {
        // ==========================================
        // 4. MATRIX MASTER PROCESS
        // ==========================================
        static int mat1[50][50], mat2[50][50], result[50][50];
        MPI_Recv(mat1, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(mat2, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Send 10 rows to each of the 5 workers (Ranks 5, 6, 7, 8, 9)
#pragma omp parallel for
        for (int dest = 5; dest <= 9; dest++) {
            int start_row = (dest - 5) * 10;
            MPI_Send(&mat1[start_row][0], 10 * 50, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&mat2[start_row][0], 10 * 50, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }

        // Receive 10 rows back from each worker
#pragma omp parallel for
        for (int source = 5; source <= 9; source++) {
            int start_row = (source - 5) * 10;
            MPI_Recv(&result[start_row][0], 10 * 50, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Send(result, 50 * 50, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank >= 5 && rank <= 9) {
        // ==========================================
        // 5. MATRIX WORKER PROCESSES (Ranks 5-9)
        // ==========================================
        int mat1_part[10][50], mat2_part[10][50], result_part[10][50];

        MPI_Recv(mat1_part, 10 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(mat2_part, 10 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#pragma omp parallel for
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 50; j++) {
                result_part[i][j] = mat1_part[i][j] + mat2_part[i][j];
            }
        }

        MPI_Send(result_part, 10 * 50, MPI_INT, 4, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
