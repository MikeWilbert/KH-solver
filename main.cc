#include "include.h"
#include "simulation.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int N = 100;

    simulation simu( N );

    MPI_Finalize();
    return 0;
}
