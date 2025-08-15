#pragma once

#include "include.h"

class simulation
{

  public:

    simulation( const size_t N_, const size_t BD_, const double cfl_ );
    ~simulation();

  private:

    int mpi_rank, mpi_size, mpi_dims[2], mpi_coords[2];
    int mpi_neighbors[8]; // W,E,S,N, NW, NE, SW, SE

    const size_t N;
    const size_t BD;
    const size_t cfl;

    const size_t N_bd;

    ArrayND<double> E;
    ArrayND<double> B;

    void init_mpi();

};