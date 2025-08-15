#include "include.h"
#include "simulation.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int N  = 100; // spatial resolution
    int BD = 2;   // # ghost cells per side

    simulation simu( N, BD );

    MPI_Finalize();
    return 0;
}
