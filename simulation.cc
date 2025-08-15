#include "simulation.h"

simulation::simulation( const int N_ ) :
N(N_), E( N_*N_, 0. ), B( N_*N_, 0. )
{

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

}

simulation::~simulation()
{



}