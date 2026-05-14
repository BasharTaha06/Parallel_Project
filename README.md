# Hybrid Parallel System (MPI + OpenMP)

This project implements a hybrid parallel computing system that combines the power of **MPI** (Message Passing Interface) for distributed-memory parallelism and **OpenMP** for shared-memory parallelism (multithreading). The program is specifically designed using a Master-Worker architecture and requires exactly **10 processes** to execute.

## 🚀 Project Description

The system coordinates 10 processes to solve different computational tasks concurrently. Each process is assigned a specific role:

* **Process 0 (Master):** The main orchestrator. It initializes variables, generates data (integers, strings, files, and matrices using OpenMP), distributes tasks to the other 4 main processes, and collects the final results.

* **Process 1 (Factorial Calculator):** Receives an integer from the Master and calculates its factorial. It utilizes OpenMP threads with a `reduction` clause to accelerate the multiplication safely.

* **Process 2 (Vowel Counter):** Receives a string and counts the total number of vowels (`a, e, i, o, u`). It uses OpenMP `parallel for` with a `reduction` clause to prevent race conditions while counting.

* **Process 3 (File Processor):** Receives a dummy text file. It reads the file and uses OpenMP `parallel sections` to split the workload: one thread writes even-indexed lines to `even.txt`, while another concurrently writes odd-indexed lines to `odd.txt`.

* **Process 4 (Matrix Manager):** Acts as a sub-master. It receives two 50x50 matrices from the Master, splits them, and distributes 10 rows to each of the 5 worker processes. Once calculated, it gathers the results and sends the final matrix back to the Master.

* **Processes 5-9 (Matrix Workers):** These five processes receive partial matrices (10 rows each) from Process 4, perform parallel matrix addition, and send their calculated parts back.

## 🛠️ Prerequisites

### Linux (Ubuntu)

To compile and run this project on a Linux environment, you need the GCC compiler and OpenMPI libraries:

```bash
sudo apt update
sudo apt install build-essential openmpi-bin libopenmpi-dev

```

### Windows

If you are using Windows, ensure you have:

1. **Visual Studio** with C++ desktop development workload installed.
2. **MS-MPI (Microsoft MPI):** Download and install both `msmpisetup.exe` (Runtime) and `msmpisdk.msi` (SDK) from the official Microsoft website.

## ⚙️ How to Run

### On Linux (Ubuntu)

1. Navigate to the project directory:
```bash
cd Parallel_Project

```


2. Compile the C++ code with OpenMP support enabled:
```bash
mpic++ -fopenmp Parallel_Project.cpp -o Parallel_Project

```


3. Execute the program using 10 processes. *(Note: If your local machine has fewer than 10 CPU cores, use the `--oversubscribe` flag to allow context-switching).*
```bash
mpirun --oversubscribe -n 10 ./Parallel_Project

```



### On Windows (Visual Studio)

1. Open the solution file (`.slnx` or `.sln`) in Visual Studio.
2. Build the project (`Ctrl + Shift + B`).
3. Open CMD or PowerShell, navigate to the output directory (e.g., `x64/Debug`), and run:
```cmd
mpiexec -n 10 Parallel_Project.exe

```



## 🧠 Key Parallel Concepts Used

* **Hybrid Parallelism:** Combining cross-process communication (MPI) with in-process multithreading (OpenMP).
* **Task Parallelism:** Distinct operations running simultaneously across different processes (e.g., Math vs. String Parsing vs. File I/O).
* **Data Parallelism:** Dividing large arrays (Matrices) into chunks and processing them concurrently.
* **Reduction:** Utilizing OpenMP's `reduction` clauses to safely aggregate data across multiple threads without explicit locks or mutexes, avoiding Race Conditions.
