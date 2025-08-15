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

  // setup 2D Cartesian processor grid
  int ndims = 2;
  mpi_dims[0] = 0; // 0 → let MPI choose
  mpi_dims[1] = 0;          
  MPI_Dims_create(mpi_size, ndims, mpi_dims);

  int periods[2] = {1, 1}; // periodic in both directions
  int reorder = 1;

  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, mpi_dims, periods, reorder, &cart_comm);

  MPI_Cart_coords(cart_comm, mpi_rank, ndims, mpi_coords);

  // get direct neighbors
  MPI_Cart_shift(cart_comm, 0, 1, &mpi_neighbors[0], &mpi_neighbors[1]); // L,R
  MPI_Cart_shift(cart_comm, 1, 1, &mpi_neighbors[2], &mpi_neighbors[3]); // U,D

  // get diagonal neighbors
  int px = mpi_dims[0];
  int py = mpi_dims[1];

  int diags[4][2] = {
      {(mpi_coords[0]-1+px)%px, (mpi_coords[1]-1+py)%py}, // TL
      {(mpi_coords[0]+1)%px,    (mpi_coords[1]-1+py)%py}, // TR
      {(mpi_coords[0]-1+px)%px, (mpi_coords[1]+1)%py},    // BL
      {(mpi_coords[0]+1)%px,    (mpi_coords[1]+1)%py}     // BR
  };

  for(int i=0; i<4; ++i) {
      MPI_Cart_rank(cart_comm, diags[i], &mpi_neighbors[4+i]);
  }

  // print info
  if(mpi_rank==0)
  { 
    std::cout << "MPI initialization complete!" << std::endl;
    std::cout << "Processor grid: " << mpi_dims[0] << "x" << mpi_dims[1] << std::endl; 
  }

}

simulation::~simulation()
{



}