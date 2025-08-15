#pragma once

#include "include.h"

class simulation
{

  public:

    simulation( const int N_  );
    ~simulation();

  private:

    int mpi_rank, mpi_size;

    const int N;
    std::vector<double> E;
    std::vector<double> B;

};