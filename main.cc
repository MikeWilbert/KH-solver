#include "include.h"
#include "simulation.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    const size_t N   = 200; // spatial resolution
    const size_t BD  = 3;   // # ghost cells per side
    const double cfl = 0.5; // CFL number

    simulation simu( N, BD, cfl );

    MPI_Finalize();
    return 0;
}
