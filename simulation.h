#pragma once

#include "include.h"

class simulation
{

  public:

    simulation( const size_t N_  );
    ~simulation();

  private:

    int mpi_rank, mpi_size;

    const size_t N;

    ArrayND<double> E;
    ArrayND<double> B;

};