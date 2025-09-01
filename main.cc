#include "include.h"
#include "simulation.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    const size_t N        = 256;     // spatial resolution
    const size_t BD       = 2;       // # ghost cells per side
    const double cfl      = 0.5;    // CFL number
    const double run_time = 3.; // run time

    simulation simu( N, BD, cfl );

    simu.run( run_time );

    MPI_Finalize();
    return 0;
}
