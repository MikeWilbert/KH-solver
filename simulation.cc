#include "simulation.h"

simulation::simulation( const size_t N_, const size_t BD_, const double cfl_ ) :
N(N_), 
BD(BD_), 
cfl(cfl_), 
N_bd(N_+2*BD_), 
E( {3, N_, N_} ),
B( {3, N_, N_} )
{

  init_mpi();

}

void simulation::init_mpi()
{

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  int ndims = 2;
  mpi_dims[0] = 0; // 0 → let MPI choose
  mpi_dims[1] = 0;          
  MPI_Dims_create(mpi_size, ndims, mpi_dims);

  int periods[2] = {1, 1}; // periodic in both directions
  int reorder = 1;

  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, mpi_dims, periods, reorder, &cart_comm);

  MPI_Cart_coords(cart_comm, mpi_rank, ndims, mpi_coords);

  if(mpi_rank==0)
  { 
    std::cout << "MPI initialization complete!" << std::endl;
    std::cout << "Processor grid: " << mpi_dims[0] << "x" << mpi_dims[1] << std::endl; 
  }

}

simulation::~simulation()
{



}