#pragma once

#include "include.h"

class simulation
{

  public:

    simulation( const size_t N_, const size_t BD_  );
    ~simulation();

  private:

    int mpi_rank, mpi_size;

    const size_t N;
    const size_t BD;
    const size_t N_bd;

    ArrayND<double> E;
    ArrayND<double> B;

};