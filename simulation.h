#pragma once

#include "include.h"

class simulation
{

  public:

    simulation( const size_t N_, const size_t BD_, const double cfl_ );
    void run( const double run_time );
    ~simulation();

  private:

    MPI_Comm cart_comm;
    int mpi_rank, mpi_size, mpi_dims[2], mpi_coords[2];
    int mpi_neighbors[8]; // W,E,S,N, NW, NE, SW, SE
    MPI_Datatype vti_subarray_scalar = MPI_DATATYPE_NULL;
    MPI_Datatype vti_subarray_vector = MPI_DATATYPE_NULL;
    MPI_Datatype vti_float3          = MPI_DATATYPE_NULL;
    float* float_array_vector;

    MPI_Datatype mpi_slice_inner_W = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_slice_inner_E = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_slice_inner_S = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_slice_inner_N = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_slice_outer_W = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_slice_outer_E = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_slice_outer_S = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_slice_outer_N = MPI_DATATYPE_NULL;

    MPI_Datatype mpi_edge_inner_SW = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_edge_inner_SE = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_edge_inner_NW = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_edge_inner_NE = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_edge_outer_SW = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_edge_outer_SE = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_edge_outer_NW = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_edge_outer_NE = MPI_DATATYPE_NULL;

    const size_t N_tot;
    const size_t BD;
    const double cfl;

    size_t N[2]; // local internal size (1D)
    size_t N_bd[2];

    size_t start_i[2]; // start index inner celles
    size_t end_i  [2]; // end   index inner celles

    double dt;
    double dx;
    double L;
    double time;

    double dx_inv;

    size_t num_outputs;

    ArrayND<double> E;
    ArrayND<double> B;
    ArrayND<double> prim_e;
    ArrayND<double> cons_e;

    ArrayND<double> E_1;
    ArrayND<double> B_1;
    ArrayND<double> RHS_BE_0;
    ArrayND<double> RHS_BE_1;

    ArrayND<double> num_flux_BE_x;
    ArrayND<double> num_flux_BE_y;

    void init_mpi();
    void setup();
    void set_ghost_cells( ArrayND<double>& field );
    void step();
    void get_dt();
    void get_RHS_BE( ArrayND<double>& RHS, const ArrayND<double>& E_, const ArrayND<double>& B_ );
    void RK_step( ArrayND<double>& E_, ArrayND<double>& B_, 
                          const ArrayND<double>& RHS_EB_1_, const ArrayND<double>& RHS_EB_2_, 
                          const double a_1, const double a_2 );

    void print_vti();
    void write_vti_header( std::string file_name, long& N_bytes_scalar, long& N_bytes_vector );
    void write_vti_footer( std::string file_name );
    void print_mpi_vector( std::string file_name, long& N_bytes_vector, const ArrayND<double>& field, const size_t comp );
    void print_mpi_scalar( std::string file_name, long& N_bytes_scalar, const ArrayND<double>& field, const size_t comp );

};