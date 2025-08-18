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

    const size_t N_tot;
    const size_t BD;
    const size_t cfl;

    size_t N[2]; // local internal size (1D)
    size_t N_bd[2];

    double dt;
    double dx;
    double L;

    ArrayND<double> E;
    ArrayND<double> B;

    void init_mpi();
    void setup();
    void print_vti();

};