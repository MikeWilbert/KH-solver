#include "simulation.h"

simulation::simulation( const size_t N_, const size_t BD_ ) :
N(N_), 
BD(BD_), 
N_bd(N_+2*BD_), 
E( {3, N_, N_} ),
B( {3, N_, N_} )
{

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::cout << "Hello from " << mpi_rank << "/" << mpi_size << std::endl;

}

simulation::~simulation()
{



}