#include "simulation.h"

simulation::simulation( const size_t N_, const size_t BD_, const double cfl_ ) :
N_tot(N_), 
BD(BD_), 
cfl(cfl_),
time(0.),
num_outputs(0)
{

  init_mpi();

  prim.resize({5, N_bd[0], N_bd[1]});
  cons.resize({5, N_bd[0], N_bd[1]});

  prim_1        .resize({5, N_bd[0], N_bd[1]});
  cons_1        .resize({5, N_bd[0], N_bd[1]});

  RHS_fluid_0     .resize({5, N[0]   , N[1]  });
  RHS_fluid_1     .resize({5, N[0]   , N[1]  });

  prim_ipol_x     .resize( {2, 5, N[0]+1, N[1]  } );
  prim_ipol_y     .resize( {2, 5, N[0]  , N[1]+1} );
  cons_ipol_x     .resize( {2, 5, N[0]+1, N[1]  } );
  cons_ipol_y     .resize( {2, 5, N[0]  , N[1]+1} );
  flux_ipol_x     .resize( {2, 5, N[0]+1, N[1]  } );
  flux_ipol_y     .resize( {2, 5, N[0]  , N[1]+1} );
  speed_ipol_x    .resize( {2,    N[0]+1, N[1]  } );
  speed_ipol_y    .resize( {2,    N[0]  , N[1]+1} );
  num_flux_fluid_x.resize( {   5, N[0]+1, N[1]  } );
  num_flux_fluid_y.resize( {   5, N[0]  , N[1]+1} );
  TVD_fluid_x     .resize( {   5, N[0]+2, N[1]  } );
  TVD_fluid_y     .resize( {   5, N[0]  , N[1]+2} );

  setup();

  if(mpi_rank==0){ std::cout << "Setup completed!" << std::endl; }

  print_vti();

}

void simulation::setup()
{
  time = 0.;

  L = 1.;
  dx = L / N_tot;
  dx_inv = 1./dx;

  double Gamma = 1.4;

  double rho, vx, vy, vz, p;

  for( size_t ix = BD; ix < N_bd[0] - BD; ix++ ){
  for( size_t iy = BD; iy < N_bd[1] - BD; iy++ ){

    double x_val = ( ix - BD + 0.5 ) * dx + mpi_coords[0] * N_tot / mpi_dims[0] * dx;
    double y_val = ( iy - BD + 0.5 ) * dx + mpi_coords[1] * N_tot / mpi_dims[1] * dx;

    // Kelvin-Helmholtz
    if( y_val > 0.75 * L || y_val < 0.25 * L ) // outside
    {

      rho = 1.;
      vx  = +0.5;
      vy  = 0.01 * sin( 2.*M_PI/L * x_val );
      vz  = 0.;
      p   = 2.5;

    }
    else // inside
    {

      rho = 2.;
      vx  = -0.5;
      vy  = 0.01 * sin( 2.*M_PI/L * x_val );
      vz  = 0.;
      p   = 2.5;

    }

    prim( 0, ix, iy ) = rho;
    prim( 1, ix, iy ) = vx;
    prim( 2, ix, iy ) = vy;
    prim( 3, ix, iy ) = vz;
    prim( 4, ix, iy ) = p;

    cons( 0, ix, iy ) = rho;
    cons( 1, ix, iy ) = rho * vx;
    cons( 2, ix, iy ) = rho * vy;
    cons( 3, ix, iy ) = rho * vz;
    cons( 4, ix, iy ) = p / ( Gamma - 1. ) + 0.5 * rho * ( vx*vx + vy*vy + vz*vz );
    
  }}

  set_ghost_cells(cons);

}

void simulation::run( const double run_time )
{

  double out_time = 0.;
  double out_interval = 0.01;

  do
  {

    step();

    time += dt;
    
    out_time += dt;
    if(out_time > out_interval)
    {
      print_vti();
      out_time -= out_interval;

    }

    if(mpi_rank==0){ std::cout << "time = " << time << std::endl; }

    // if(mpi_rank==0){ std::cout << "\rSimulation time: " << time << "   " << std::flush; }

  } while ( time < run_time );

  if(mpi_rank==0){ std::cout << std::endl; }

}

void simulation::get_dt()
{

  // fluid cfl
  double v_max = 0.;
  double v_max_loc = 0.;

  for( size_t ix = BD; ix < N_bd[0]-BD; ix++ ){
  for( size_t iy = BD; iy < N_bd[1]-BD; iy++ ){

    double Gamma = 1.4;

    double rho = prim(0, ix, iy);
    double vx  = prim(1, ix, iy);
    double vy  = prim(2, ix, iy);
    double vz  = prim(3, ix, iy);
    double p   = prim(4, ix, iy);

    double c_s = sqrt( Gamma * p / rho );
    double v2  = vx*vx + vy*vy + vz*vz;
    
    double v_tmp = sqrt(v2) + c_s;

    v_max_loc = std::max( v_max_loc, v_tmp );

  }}

  MPI_Allreduce( &v_max_loc, &v_max, 1, MPI_DOUBLE, MPI_SUM,cart_comm );

  dt = cfl * dx / v_max;

}

void simulation::RK_step( ArrayND<double>& cons_,
                          const ArrayND<double>& RHS_fluid_1_, const ArrayND<double>& RHS_fluid_2_,
                          const double a_1, const double a_2 )
{

  for( size_t ix = 0; ix < N[0]; ix++ ){
  for( size_t iy = 0; iy < N[1]; iy++ ){

    size_t jx = ix+BD;
    size_t jy = iy+BD;

    cons_(0, jx, jy) = cons(0, jx, jy) + a_1 * dt * RHS_fluid_1_( 0, ix, iy ) + a_2 * dt * RHS_fluid_2_( 0, ix, iy );
    cons_(1, jx, jy) = cons(1, jx, jy) + a_1 * dt * RHS_fluid_1_( 1, ix, iy ) + a_2 * dt * RHS_fluid_2_( 1, ix, iy );
    cons_(2, jx, jy) = cons(2, jx, jy) + a_1 * dt * RHS_fluid_1_( 2, ix, iy ) + a_2 * dt * RHS_fluid_2_( 2, ix, iy );
    cons_(3, jx, jy) = cons(3, jx, jy) + a_1 * dt * RHS_fluid_1_( 3, ix, iy ) + a_2 * dt * RHS_fluid_2_( 3, ix, iy );
    cons_(4, jx, jy) = cons(4, jx, jy) + a_1 * dt * RHS_fluid_1_( 4, ix, iy ) + a_2 * dt * RHS_fluid_2_( 4, ix, iy );

  }}

  set_ghost_cells(cons_);

}

void simulation::get_primitives( const ArrayND<double>& cons )
{

  // get primitives
  for( size_t ix = 0; ix < N_bd[0]; ix++ ){
  for( size_t iy = 0; iy < N_bd[1]; iy++ ){

    double Gamma = 1.4;

    double rho = cons(0, ix, iy);
    double vx  = cons(1, ix, iy) / rho;
    double vy  = cons(2, ix, iy) / rho;
    double vz  = cons(3, ix, iy) / rho;
    double p   = ( Gamma - 1. ) * ( cons(4, ix, iy) - 0.5 * rho * ( vx*vx + vy*vy + vz*vz ) );

    prim(0, ix, iy) = rho;
    prim(1, ix, iy) = vx;
    prim(2, ix, iy) = vy;
    prim(3, ix, iy) = vz;
    prim(4, ix, iy) = p;

  }}

}

template <typename T> constexpr int simulation::sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

double simulation::minmod( const double a, const double b )
{

  double result = 0.5 * ( sgn(a) + sgn(b) ) * std::min( fabs(a), fabs(b) );

  return result;

}

double simulation::minmod( const double a, const double b, const double c )
{

  double result = 0.25 * ( sgn(a) + sgn(b) ) * ( sgn(a) + sgn(c) ) * sgn(a) * std::min( fabs(a), fabs(b) );

  return result;

}

inline double simulation::sqr(double x) {
    return x * x;
}

void simulation::reconstruct( const ArrayND<double>& prim     , ArrayND<double>& TVD_fluid, ArrayND<double>& prim_ipol, 
                                    ArrayND<double>& cons_ipol, ArrayND<double>& flux_ipol, ArrayND<double>& speed_ipol,
                              int dir )
{

  // dir:0 -> x
  // dir:1 -> y

  size_t end[2]   = { N[0], N[1] } ;
  end[dir] += 1;

  int    shift[2] = { 1-dir, 0+dir }; // dir:0 -> (1, 0) , dir:1 -> ( 0, 1)

  // interpolate

  // interplation from the LEFT
  for( size_t i  = 0; i  < 5   ; i++ ){
  for( size_t ix = 0; ix < end[0]; ix++ ){
  for( size_t iy = 0; iy < end[1]; iy++ ){

    // index of the cell center to interplate from
    size_t jx = ix+BD-shift[0];
    size_t jy = iy+BD-shift[1];

    double prim_l3 = prim( i, jx-2*shift[0], jy-2*shift[1] ); // q_{j-2}
    double prim_l2 = prim( i, jx-  shift[0], jy-  shift[1] ); // q_{j-1}
    double prim_l1 = prim( i, jx           , jy            ); // q_{j  }
    double prim_r1 = prim( i, jx+  shift[0], jy+  shift[1] ); // q_{j+1}
    double prim_r2 = prim( i, jx+2*shift[0], jy+2*shift[1] ); // q_{j+2}

    // all possible interpolations with order 3
    double inter_1 =  1./3.*prim_l3 - 7./6.*prim_l2 + 11./6.*prim_l1;
    double inter_2 = -1./6.*prim_l2 + 5./6.*prim_l1 +  1./3.*prim_r1;
    double inter_3 =  1./3.*prim_l1 + 5./6.*prim_r1 -  1./6.*prim_r2;

    double gamma_1 = 0.1;
    double gamma_2 = 0.6;
    double gamma_3 = 0.3;

    double beta_1 = 13./12. * sqr( prim_l3 - 2.*prim_l2 + prim_l1 ) + 1./4. * sqr(    prim_l3 - 4.*prim_l2 + 3.*prim_l1 );
    double beta_2 = 13./12. * sqr( prim_l2 - 2.*prim_l1 + prim_r1 ) + 1./4. * sqr(    prim_l2              -    prim_r1 );
    double beta_3 = 13./12. * sqr( prim_l1 - 2.*prim_r1 + prim_r2 ) + 1./4. * sqr( 3.*prim_l1 - 4.*prim_r1 +    prim_r2 );

    double eps = 1.e-6;

    double alpha_1 = gamma_1 / sqr( eps + beta_1 ); 
    double alpha_2 = gamma_2 / sqr( eps + beta_2 );
    double alpha_3 = gamma_3 / sqr( eps + beta_3 );

    double alpha_sum_inv = 1. / ( alpha_1 + alpha_2 + alpha_3 );

    double omega_1 = alpha_1 * alpha_sum_inv;
    double omega_2 = alpha_2 * alpha_sum_inv;
    double omega_3 = alpha_3 * alpha_sum_inv;

    prim_ipol( 0, i, ix, iy ) = omega_1 * inter_1 + omega_2 * inter_2 + omega_3 * inter_3;

  }}}


  // interplation from the RIGHT
  for( size_t i  = 0; i  < 5   ; i++ ){
  for( size_t ix = 0; ix < end[0]; ix++ ){
  for( size_t iy = 0; iy < end[1]; iy++ ){

    // index of the cell center to interplate from
    size_t jx = ix+BD-shift[0];
    size_t jy = iy+BD-shift[1];

    double prim_l2 = prim( i, jx-  shift[0], jy-  shift[1] ); // q_{j-1}
    double prim_l1 = prim( i, jx           , jy            ); // q_{j  }
    double prim_r1 = prim( i, jx+  shift[0], jy+  shift[1] ); // q_{j+1}
    double prim_r2 = prim( i, jx+2*shift[0], jy+2*shift[1] ); // q_{j+2}
    double prim_r3 = prim( i, jx+3*shift[0], jy+3*shift[1] ); // q_{j+3}

    // all possible interpolations with order 3
    double inter_1 =  1./3.*prim_r3 - 7./6.*prim_r2 + 11./6.*prim_r1;
    double inter_2 = -1./6.*prim_r2 + 5./6.*prim_r1 +  1./3.*prim_l1;
    double inter_3 =  1./3.*prim_r1 + 5./6.*prim_l1 -  1./6.*prim_l2;

    double gamma_1 = 0.1;
    double gamma_2 = 0.6;
    double gamma_3 = 0.3;

    double beta_1 = 13./12. * sqr( prim_r3 - 2.*prim_r2 + prim_r1 ) + 1./4. * sqr(    prim_r3 - 4.*prim_r2 + 3.*prim_r1 );
    double beta_2 = 13./12. * sqr( prim_r2 - 2.*prim_r1 + prim_l1 ) + 1./4. * sqr(    prim_r2              -    prim_l1 );
    double beta_3 = 13./12. * sqr( prim_r1 - 2.*prim_l1 + prim_l2 ) + 1./4. * sqr( 3.*prim_r1 - 4.*prim_l1 +    prim_l2 );

    double eps = 1.e-6;

    double alpha_1 = gamma_1 / sqr( eps + beta_1 ); 
    double alpha_2 = gamma_2 / sqr( eps + beta_2 );
    double alpha_3 = gamma_3 / sqr( eps + beta_3 );

    double alpha_sum_inv = 1. / ( alpha_1 + alpha_2 + alpha_3 );

    double omega_1 = alpha_1 * alpha_sum_inv;
    double omega_2 = alpha_2 * alpha_sum_inv;
    double omega_3 = alpha_3 * alpha_sum_inv;

    prim_ipol( 1, i, ix, iy ) = omega_1 * inter_1 + omega_2 * inter_2 + omega_3 * inter_3;

  }}}

  // physical flux and max absolute speeds
  for( size_t is = 0; is < 2     ; is++ ){
  for( size_t ix = 0; ix < end[0]; ix++ ){
  for( size_t iy = 0; iy < end[1]; iy++ ){

    double Gamma = 1.4;

    double rho = prim_ipol(is, 0, ix, iy);
    double vx  = prim_ipol(is, 1, ix, iy);
    double vy  = prim_ipol(is, 2, ix, iy);
    double vz  = prim_ipol(is, 3, ix, iy);
    double p   = prim_ipol(is, 4, ix, iy);

    double v_dir = prim_ipol(is, 1+dir, ix, iy);

    double E   = p / ( Gamma - 1. ) + 0.5 * rho * ( vx*vx + vy*vy + vz*vz );

    cons_ipol( is, 0, ix, iy ) = rho;
    cons_ipol( is, 1, ix, iy ) = rho * vx;
    cons_ipol( is, 2, ix, iy ) = rho * vy;
    cons_ipol( is, 3, ix, iy ) = rho * vz;
    cons_ipol( is, 4, ix, iy ) = E;

    flux_ipol( is, 0, ix, iy ) = v_dir * rho;
    flux_ipol( is, 1, ix, iy ) = v_dir * vx * rho;
    flux_ipol( is, 2, ix, iy ) = v_dir * vy * rho;
    flux_ipol( is, 3, ix, iy ) = v_dir * vz * rho;
    flux_ipol( is, 4, ix, iy ) = v_dir * ( E + p );

    flux_ipol( is, 1+dir, ix, iy ) += p;

    double c_s = sqrt( Gamma * p / rho );

    speed_ipol( is, ix, iy ) = fabs( v_dir ) + c_s;

  }}}

}

void simulation::get_num_flux( ArrayND<double>& num_flux_fluid, const ArrayND<double>& flux_ipol, 
                      const ArrayND<double>& cons_ipol, const ArrayND<double>& speed_ipol, int dir )
{
  // dir:0 -> x
  // dir:1 -> y

  size_t end[2]   = { N[0], N[1] } ;
  end[dir] += 1;

  for( size_t i = 0; i < 5; i++  )
  {

    for( size_t ix = 0; ix < end[0]; ix++ ){
    for( size_t iy = 0; iy < end[1]; iy++ ){

      double max_speed = std::max( speed_ipol( 1, ix, iy ), speed_ipol( 0, ix, iy ) );

      num_flux_fluid( i, ix, iy ) =             0.5 * ( flux_ipol( 1, i, ix, iy) + flux_ipol( 0, i, ix, iy) )
                                  - max_speed * 0.5 * ( cons_ipol( 1, i, ix, iy) - cons_ipol( 0, i, ix, iy) );

    }}

  }

}

void simulation::get_RHS_fluid( ArrayND<double>& RHS, ArrayND<double>& cons )
{

  // primitives
  get_primitives( cons );

  // interpolate
  reconstruct( prim, TVD_fluid_x, prim_ipol_x, cons_ipol_x, flux_ipol_x, speed_ipol_x, 0 );
  reconstruct( prim, TVD_fluid_y, prim_ipol_y, cons_ipol_y, flux_ipol_y, speed_ipol_y, 1 );

  // numerical flux
  get_num_flux( num_flux_fluid_x, flux_ipol_x, cons_ipol_x, speed_ipol_x, 0 );
  get_num_flux( num_flux_fluid_y, flux_ipol_y, cons_ipol_y, speed_ipol_y, 1 );

  // RHS
  for( size_t i = 0; i < 5; i++  )
  {

    for( size_t ix = 0; ix < N[0]; ix++ ){
    for( size_t iy = 0; iy < N[1]; iy++ ){

      RHS( i, ix, iy ) = - ( num_flux_fluid_x( i, ix+1, iy   ) - num_flux_fluid_x( i, ix, iy ) ) * dx_inv
                         - ( num_flux_fluid_y( i, ix  , iy+1 ) - num_flux_fluid_y( i, ix, iy ) ) * dx_inv;

    }}

  }

}

void simulation::step()
{

  get_dt();

  get_RHS_fluid( RHS_fluid_0, cons );
  RK_step      ( cons_1, RHS_fluid_0, RHS_fluid_1, 1. , 0.  );

  get_RHS_fluid( RHS_fluid_1, cons_1 );
  RK_step      ( cons  , RHS_fluid_0, RHS_fluid_1, 0.5, 0.5 );

  get_primitives( cons );

}

void simulation::set_ghost_cells( ArrayND<double>& field )
{

  size_t num_fields = field.dim_size( 0 );

  // exchange ghost cells for each component
  for( size_t i = 0; i < num_fields; i++ )
  {

    double* base = &field( i, 0, 0 );
    int tag = 123;

    // W -> E
    MPI_Sendrecv( base, 1, mpi_slice_inner_W, mpi_neighbors[0], tag,
                  base, 1, mpi_slice_outer_E, mpi_neighbors[1], tag, cart_comm, MPI_STATUS_IGNORE);

    // E -> W
    MPI_Sendrecv( base, 1, mpi_slice_inner_E, mpi_neighbors[1], tag,
                  base, 1, mpi_slice_outer_W, mpi_neighbors[0], tag, cart_comm, MPI_STATUS_IGNORE);

    // S -> N
    MPI_Sendrecv( base, 1, mpi_slice_inner_S, mpi_neighbors[2], tag,
                  base, 1, mpi_slice_outer_N, mpi_neighbors[3], tag, cart_comm, MPI_STATUS_IGNORE);

    // N -> S
    MPI_Sendrecv( base, 1, mpi_slice_inner_N, mpi_neighbors[3], tag,
                  base, 1, mpi_slice_outer_S, mpi_neighbors[2], tag, cart_comm, MPI_STATUS_IGNORE);

    // SW -> NE
    MPI_Sendrecv( base, 1, mpi_edge_inner_SW, mpi_neighbors[6], tag,
                  base, 1, mpi_edge_outer_NE, mpi_neighbors[5], tag, cart_comm, MPI_STATUS_IGNORE);

    // NE -> SW
    MPI_Sendrecv( base, 1, mpi_edge_inner_NE, mpi_neighbors[5], tag,
                  base, 1, mpi_edge_outer_SW, mpi_neighbors[6], tag, cart_comm, MPI_STATUS_IGNORE);

    // SE -> NW
    MPI_Sendrecv( base, 1, mpi_edge_inner_SE, mpi_neighbors[7], tag,
                  base, 1, mpi_edge_outer_NW, mpi_neighbors[4], tag, cart_comm, MPI_STATUS_IGNORE);

    // NW -> SE
    MPI_Sendrecv( base, 1, mpi_edge_inner_NW, mpi_neighbors[4], tag,
                  base, 1, mpi_edge_outer_SE, mpi_neighbors[7], tag, cart_comm, MPI_STATUS_IGNORE);

  }

}

void simulation::print_vti()
{

  const std::string file_name = "/p/scratch/specturb/KHI/N_1152/output_" + std::to_string(num_outputs) + ".vti";

  long N_bytes_scalar, N_bytes_vector;

  write_vti_header( file_name, N_bytes_scalar, N_bytes_vector );

  // print_mpi_vector( file_name, N_bytes_vector, prim, 1 );
  print_mpi_scalar( file_name, N_bytes_scalar, prim, 0 );
  // print_mpi_scalar( file_name, N_bytes_scalar, prim, 4 );

  write_vti_footer( file_name );

  num_outputs += 1;

}

void simulation::print_mpi_vector( std::string file_name, long& N_bytes_vector, const ArrayND<double>& field, const size_t comp )
{
  if(mpi_rank==0)
  {
    std::ofstream binary_os(file_name.c_str(), std::ios::out | std::ios::app | std::ios::binary );
    binary_os.write(reinterpret_cast<const char*>(&N_bytes_vector),sizeof(uint64_t)); // size of following binary package
    binary_os.close();
  }MPI_Barrier(cart_comm);
  
  // open file
  MPI_File mpi_file;
  MPI_File_open(cart_comm, file_name.c_str(), MPI_MODE_APPEND|MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_file);
  
  // offset to end of file
  MPI_Offset mpi_eof;
  MPI_File_get_position(mpi_file, &mpi_eof);
  MPI_Barrier(cart_comm);
  
  for( size_t iy = BD; iy < N_bd[1] - BD; iy++ ){
  for( size_t ix = BD; ix < N_bd[0] - BD; ix++ ){
  
    size_t id = (iy-BD) * N[0] + (ix-BD);
    
    float_array_vector[3*id+0] = float( field(0+comp,ix,iy) );
    float_array_vector[3*id+1] = float( field(1+comp,ix,iy) );
    float_array_vector[3*id+2] = float( field(2+comp,ix,iy) ); 

  }}
  
  // write data
  MPI_File_set_view(mpi_file, mpi_eof, vti_float3, vti_subarray_vector, "native", MPI_INFO_NULL);
  MPI_File_write_all(mpi_file, float_array_vector, N[0]*N[1], vti_float3, MPI_STATUS_IGNORE);
  
  // close file
  MPI_File_close(&mpi_file);  
}

void simulation::print_mpi_scalar( std::string file_name, long& N_bytes_scalar, const ArrayND<double>& field, const size_t comp )
{
  if(mpi_rank==0)
  {
    std::ofstream binary_os(file_name.c_str(), std::ios::out | std::ios::app | std::ios::binary );
    binary_os.write(reinterpret_cast<const char*>(&N_bytes_scalar),sizeof(uint64_t)); // size of following binary package
    binary_os.close();
  }MPI_Barrier(cart_comm);
  
  // open file
  MPI_File mpi_file;
  MPI_File_open(cart_comm, file_name.c_str(), MPI_MODE_APPEND|MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_file);
  
  // offset to end of file
  MPI_Offset mpi_eof;
  MPI_File_get_position(mpi_file, &mpi_eof);
  MPI_Barrier(cart_comm);
  
  for( size_t iy = BD; iy < N_bd[1] - BD; iy++ ){
  for( size_t ix = BD; ix < N_bd[0] - BD; ix++ ){
    
    size_t id = (iy-BD) * N[0] + (ix-BD);
    
    float_array_vector[id] = float( field(comp,ix,iy) );

  }}
  
  // write data
  MPI_File_set_view(mpi_file, mpi_eof, MPI_FLOAT, vti_subarray_scalar, "native", MPI_INFO_NULL);
  MPI_File_write_all(mpi_file, float_array_vector, N[0]*N[1], MPI_FLOAT, MPI_STATUS_IGNORE);
  
  // close file
  MPI_File_close(&mpi_file);  
}

void simulation::write_vti_header( std::string file_name, long& N_bytes_scalar, long& N_bytes_vector )
{

  std::ofstream os;
  
  long N_l = N_tot;
  long offset = 0;
	long N_all = N_l*N_l;
	     N_bytes_scalar  =   N_all * sizeof(float);
	     N_bytes_vector  = 3*N_all * sizeof(float);
  long bin_size_scalar = N_bytes_scalar + sizeof(uint64_t);// 2nd term is the size of the the leading integer announcing the numbers n the data chunk
  long bin_size_vector = N_bytes_vector + sizeof(uint64_t);

  // header
  if(mpi_rank==0)
  {
    os.open(file_name.c_str(), std::ios::out);
    if(!os){
      std::cout << "Cannot write vti header to file '" << file_name << "'!\n";
    }
    
    // write header	
		int extend_l[2]  = {0, 0};
		int extend_r[2]  = {static_cast<int>(N_tot), static_cast<int>(N_tot)};
		double origin[3] = {0.,0.,0.};
    
    os << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">" << std::endl;	
    os << "  <ImageData WholeExtent=\"" << extend_l[0] << " " << extend_r[0] << " " 
                                        << extend_l[1] << " " << extend_r[1] << " " 
                                        << "0" << " " << "1"
				 << "\" Origin=\""  << origin[0]  << " " << origin[1]  << " " << origin[2] 
				 << "\" Spacing=\"" << dx << " " << dx << " " << dx
         << "\" Direction=\"1 0 0 0 1 0 0 0 1\">" << std::endl;
    
    os << "      <FieldData>" << std::endl;
    os << "        <DataArray type=\"Float32\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">" << std::endl;
    os << "        "<< float(time) << std::endl;
    os << "        </DataArray>" << std::endl;
    os << "      </FieldData>" << std::endl;
        
		os << "    <Piece Extent=\"" << extend_l[0] << " " << extend_r[0] << " " 
                                 << extend_l[1] << " " << extend_r[1] << " " 
                                 << "0" << " " << "1" << "\">" << std::endl;
    
    os << "      <CellData>" << std::endl;
    // os << "        <DataArray type=\"Float32\" Name=\"V\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << offset << "\">" << std::endl;
    // os << "        </DataArray>" << std::endl;
    // offset += bin_size_vector;
    os << "        <DataArray type=\"Float32\" Name=\"rho\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << offset << "\">" << std::endl;
    os << "        </DataArray>" << std::endl;
    offset += bin_size_scalar;
    // os << "        <DataArray type=\"Float32\" Name=\"p\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << offset << "\">" << std::endl;
    // os << "        </DataArray>" << std::endl;
    // offset += bin_size_scalar;
    os << "      </CellData>" << std::endl;
    os << "      <PointData>" << std::endl;
    os << "      </PointData>" << std::endl;
    os << "    </Piece>" << std::endl;
    os << "  </ImageData>" << std::endl;
    os << "  <AppendedData encoding=\"raw\">" << std::endl;
    os << "   _";
                                
    os.close();
  
  }MPI_Barrier(cart_comm);

}

void simulation::write_vti_footer( std::string file_name )
{

  std::ofstream os;

  // footer
  if(mpi_rank==0)
  {
    os.open(file_name.c_str(), std::ios::out | std::ios::app);
    if(!os){
      std::cout << "Cannot write footer to file '" << file_name << "'!\n";
      exit(3);
    }
		
		os << std::endl << "  </AppendedData>" << std::endl;
    os<< "</VTKFile>" << std::endl;
	
    os.close();
  }MPI_Barrier(cart_comm);

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

  MPI_Cart_create(MPI_COMM_WORLD, ndims, mpi_dims, periods, reorder, &cart_comm);

  MPI_Comm_rank  (cart_comm, &mpi_rank);
  MPI_Comm_size  (cart_comm, &mpi_size);
  MPI_Cart_coords(cart_comm, mpi_rank, ndims, mpi_coords);

  // get direct neighbors
  MPI_Cart_shift(cart_comm, 0, 1, &mpi_neighbors[0], &mpi_neighbors[1]); // L,R
  MPI_Cart_shift(cart_comm, 1, 1, &mpi_neighbors[2], &mpi_neighbors[3]); // U,D

  // get diagonal neighbors
  int px = mpi_dims[0];
  int py = mpi_dims[1];

  int diags[4][2] = {
      {(mpi_coords[0]-1+px)%px, (mpi_coords[1]-1+py)%py}, // TL
      {(mpi_coords[0]+1   )%px, (mpi_coords[1]-1+py)%py}, // TR
      {(mpi_coords[0]-1+px)%px, (mpi_coords[1]+1)   %py}, // BL
      {(mpi_coords[0]+1   )%px, (mpi_coords[1]+1)   %py}  // BR
  };

  for(int i=0; i<4; ++i) {
      MPI_Cart_rank(cart_comm, diags[i], &mpi_neighbors[4+i]);
  }

  // check if total resolution is dividable by processor dimensions
  if ( N_tot % mpi_dims[0] != 0 || N_tot % mpi_dims[1] != 0 ) {
    std::cerr << "Spatial resolution is not dividable by processor dimensions!\n";
    std::cerr << "N     = " << N_tot << std::endl; 
    std::cerr << "pdims = " << mpi_dims[0] << "x" << mpi_dims[1] << std::endl; 
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

  // set local sizes
  N[0] = N_tot / mpi_dims[0];
  N[1] = N_tot / mpi_dims[1];

  N_bd[0] = N[0] + 2*BD;
  N_bd[1] = N[1] + 2*BD;

  // create subarrays for ghost cell exchange
  int sizes   [2];
  int subsizes[2];
  int starts  [2];
  int order = MPI_ORDER_C;
  MPI_Datatype type  = MPI_DOUBLE;

  sizes[0] = N_bd[0];
  sizes[1] = N_bd[1];

  // faces

  // West/East
  subsizes[0] = BD;
  subsizes[1] = N[1];
  starts  [1] = BD;

  // West - outer
  starts[0] = 0;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_slice_outer_W );
  MPI_Type_commit(&mpi_slice_outer_W);

  // West - inner
  starts[0] = BD;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_slice_inner_W );
  MPI_Type_commit(&mpi_slice_inner_W);

  // East - inner
  starts[0] = N[0];

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_slice_inner_E );
  MPI_Type_commit(&mpi_slice_inner_E);

  // East - outer
  starts[0] = N[0] + BD;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_slice_outer_E );
  MPI_Type_commit(&mpi_slice_outer_E);

  // Soust/North
  subsizes[0] = N[0];
  subsizes[1] = BD;
  starts  [0] = BD;

  // West - outer
  starts[1] = 0;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_slice_outer_S );
  MPI_Type_commit(&mpi_slice_outer_S);

  // West - inner
  starts[1] = BD;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_slice_inner_S );
  MPI_Type_commit(&mpi_slice_inner_S);

  // East - inner
  starts[1] = N[1];

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_slice_inner_N );
  MPI_Type_commit(&mpi_slice_inner_N);

  // East - outer
  starts[1] = N[1] + BD;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_slice_outer_N );
  MPI_Type_commit(&mpi_slice_outer_N);

  // edges
  subsizes[0] = BD;
  subsizes[1] = BD;

  // SW - inner
  starts[0] = BD;
  starts[1] = BD;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_edge_inner_SW );
  MPI_Type_commit(&mpi_edge_inner_SW);

  // SW - outer
  starts[0] = 0;
  starts[1] = 0;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_edge_outer_SW );
  MPI_Type_commit(&mpi_edge_outer_SW);

  // SE - inner
  starts[0] = N[0];
  starts[1] = BD;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_edge_inner_SE );
  MPI_Type_commit(&mpi_edge_inner_SE);

  // SE - outer
  starts[0] = N[0] + BD;
  starts[1] = 0;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_edge_outer_SE );
  MPI_Type_commit(&mpi_edge_outer_SE);

  // NW - inner
  starts[0] = BD;
  starts[1] = N[1];

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_edge_inner_NW );
  MPI_Type_commit(&mpi_edge_inner_NW);

  // NW - outer
  starts[0] = 0;
  starts[1] = N[1] + BD;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_edge_outer_NW );
  MPI_Type_commit(&mpi_edge_outer_NW);

  // NE - inner
  starts[0] = N[0];
  starts[1] = N[1];

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_edge_inner_NE );
  MPI_Type_commit(&mpi_edge_inner_NE);

  // NE - outer
  starts[0] = N[0] + BD;
  starts[1] = N[1] + BD;

  MPI_Type_create_subarray( 2, sizes, subsizes, starts, order, type, &mpi_edge_outer_NE );
  MPI_Type_commit(&mpi_edge_outer_NE);

  // create derived MPI_datatypes for vti output ( sizes swapped because of Fortran order in vtk! )
  int size_total[3] = { static_cast<int>(N_tot)             ,static_cast<int>(N_tot             ),1 };
  int mpi_start[3]  = { static_cast<int>(mpi_coords[1]*N[1]),static_cast<int>(mpi_coords[0]*N[0]),0 };
  int mpi_size[3]   = { static_cast<int>(              N[1]),static_cast<int>(              N[0]),1 };

  MPI_Type_contiguous(3, MPI_FLOAT, &vti_float3);
  MPI_Type_commit(&vti_float3);

  MPI_Type_create_subarray(3, size_total, mpi_size, mpi_start, MPI_ORDER_C, MPI_FLOAT, &vti_subarray_scalar);
  MPI_Type_commit(&vti_subarray_scalar);
  
  MPI_Type_create_subarray(3, size_total,mpi_size, mpi_start, MPI_ORDER_C, vti_float3, &vti_subarray_vector);
  MPI_Type_commit(&vti_subarray_vector);

  float_array_vector = new float[ 3*N[0]*N[1] ];

  // print info
  if(mpi_rank==0)
  { 
    std::cout << "MPI initialization complete!" << std::endl;
    std::cout << "Processor grid: " << mpi_dims[0] << "x" << mpi_dims[1] << std::endl; 
  }

}

simulation::~simulation()
{

  delete[] float_array_vector;

}