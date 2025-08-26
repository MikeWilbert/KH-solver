#include "simulation.h"

simulation::simulation( const size_t N_, const size_t BD_, const double cfl_ ) :
N_tot(N_), 
BD(BD_), 
cfl(cfl_),
time(0.),
num_outputs(0)
{

  init_mpi();

  E.resize({3, N_bd[0], N_bd[1]});
  B.resize({3, N_bd[0], N_bd[1]});

  B_1.resize({3, N_bd[0], N_bd[1]});
  E_1.resize({3, N_bd[0], N_bd[1]});

  RHS_BE_0     .resize({6, N[0]   , N[1]   });
  RHS_BE_1     .resize({6, N[0]   , N[1]   });
  num_flux_BE_x.resize({6, N[0]+1 , N[1]  });
  num_flux_BE_y.resize({6, N[0]   , N[1]+1});

  setup();

  print_vti();

}

void simulation::setup()
{
  time = 0.;

  L = 2.*M_PI;
  dx = L / N_tot;
  dx_inv = 1./dx;

  int kx = 2;
  int ky = 2;

  for( size_t ix = start_i[0]; ix < end_i[0]; ix++ ){
  for( size_t iy = start_i[1]; iy < end_i[1]; iy++ ){

    double x_val = ( ix - BD + 0.5 ) * dx + mpi_coords[0] * N_tot / mpi_dims[0] * dx;
    double y_val = ( iy - BD + 0.5 ) * dx + mpi_coords[1] * N_tot / mpi_dims[1] * dx;

    // mono chromatic wave
    E(0,ix,iy) = 0.;
    E(1,ix,iy) = sin( kx * x_val + ky * y_val );  // Ey
    E(2,ix,iy) = 0.;

    B(0,ix,iy) = 0.;
    B(1,ix,iy) = 0.;
    B(2,ix,iy) = sin( kx * x_val + ky * y_val );  // Bz

  }}

  set_ghost_cells(E);
  set_ghost_cells(B);

}

void simulation::run( const double run_time )
{

  double out_time = 0.;
  double out_interval = 0.2;

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


    if(mpi_rank==0){ std::cout << "\rSimulation time: " << time << "   " << std::flush; }



  } while ( time < run_time );

  if(mpi_rank==0){ std::cout << std::endl; }

}

void simulation::get_dt()
{

  dt = cfl * dx;

}

void simulation::get_RHS_BE( ArrayND<double>& RHS, const ArrayND<double>& E_, const ArrayND<double>& B_ )
{

  ArrayND<double> Ez_tilde( { N[0]+1, N[1]+1 } );
  ArrayND<double> Bz_tilde( { N[0]+1, N[1]+1 } );

  double E_SW[3];
  double E_SE[3];
  double E_NW[3];
  double E_NE[3];

  double B_SW[3];
  double B_SE[3];
  double B_NW[3];
  double B_NE[3];

  for( size_t ix = 0; ix < N[0]+1; ix++ ){
  for( size_t iy = 0; iy < N[1]+1; iy++ ){

      size_t jx = ix+BD;
      size_t jy = iy+BD;

      for( size_t i = 0; i < 3; i++ )
      {

        E_SW[i] = E_( i, jx-1, jy-1 );
        E_SE[i] = E_( i, jx  , jy-1 );
        E_NW[i] = E_( i, jx-1, jy   );
        E_NE[i] = E_( i, jx  , jy   );

        B_SW[i] = B_( i, jx-1, jy-1 );
        B_SE[i] = B_( i, jx  , jy-1 );
        B_NW[i] = B_( i, jx-1, jy   );
        B_NE[i] = B_( i, jx  , jy   );

      }

      double Ez_avg = 0.25 * ( E_SW[2] + E_SE[2] + E_NE[2] + E_NW[2] );
      double Bz_avg = 0.25 * ( B_SW[2] + B_SE[2] + B_NE[2] + B_NW[2] );

      double Bx_N = 0.5 * ( B_NW[0] + B_NE[0] );
      double Bx_S = 0.5 * ( B_SW[0] + B_SE[0] );
      double By_W = 0.5 * ( B_SW[1] + B_NW[1] );
      double By_E = 0.5 * ( B_SE[1] + B_NE[1] );

      double Ex_N = 0.5 * ( E_NW[0] + E_NE[0] );
      double Ex_S = 0.5 * ( E_SW[0] + E_SE[0] );
      double Ey_W = 0.5 * ( E_SW[1] + E_NW[1] );
      double Ey_E = 0.5 * ( E_SE[1] + E_NE[1] );

      Ez_tilde( ix,iy ) = Ez_avg + 0.5 * ( Bx_N - Bx_S ) + 0.5 * ( By_W - By_E ); 
      Bz_tilde( ix,iy ) = Bz_avg + 0.5 * ( Ex_N - Ex_S ) + 0.5 * ( Ey_W - Ey_E ); 

  }}

  for( size_t ix = 0; ix < N[0]+1; ix++ ){
  for( size_t iy = 0; iy < N[1]  ; iy++ ){

    size_t jx = ix+BD;
    size_t jy = iy+BD;

    double Ey_mns = E_( 1, jx-1, jy );
    double Ey_pls = E_( 1, jx  , jy );
    double Ez_mns = E_( 2, jx-1, jy );
    double Ez_pls = E_( 2, jx  , jy );

    double By_mns = B_( 1, jx-1, jy );
    double By_pls = B_( 1, jx  , jy );
    double Bz_mns = B_( 2, jx-1, jy );
    double Bz_pls = B_( 2, jx  , jy );

    num_flux_BE_x( 0, ix, iy ) =   0.;
    num_flux_BE_x( 1, ix, iy ) = - 0.5 * ( Ez_tilde( ix, iy+1 ) + Ez_tilde( ix, iy ) );
    num_flux_BE_x( 2, ix, iy ) = + 0.5 * ( Ey_pls + Ey_mns ) - 0.5 * ( Bz_pls - Bz_mns );

    num_flux_BE_x( 3, ix, iy ) =   0.;
    num_flux_BE_x( 4, ix, iy ) = + 0.5 * ( Bz_tilde( ix, iy+1 ) + Bz_tilde( ix, iy ) );
    num_flux_BE_x( 5, ix, iy ) = - 0.5 * ( By_pls + By_mns ) - 0.5 * ( Ez_pls - Ez_mns );

  }}

  for( size_t ix = 0; ix < N[0]  ; ix++ ){
  for( size_t iy = 0; iy < N[1]+1; iy++ ){

    size_t jx = ix+BD;
    size_t jy = iy+BD;

    double Ex_mns = E( 0, jx, jy-1 );
    double Ex_pls = E( 0, jx, jy   );
    double Ez_mns = E( 2, jx, jy-1 );
    double Ez_pls = E( 2, jx, jy   );

    double Bx_mns = B( 0, jx, jy-1 );
    double Bx_pls = B( 0, jx, jy   );
    double Bz_mns = B( 2, jx, jy-1 );
    double Bz_pls = B( 2, jx, jy   );
    
    num_flux_BE_y( 0, ix, iy ) = + 0.5 * ( Ez_tilde( ix+1, iy ) + Ez_tilde( ix, iy ) );
    num_flux_BE_y( 1, ix, iy ) =   0.;
    num_flux_BE_y( 2, ix, iy ) = - 0.5 * ( Ex_pls + Ex_mns ) - 0.5 * ( Bz_pls - Bz_mns );

    num_flux_BE_y( 3, ix, iy ) = - 0.5 * ( Bz_tilde( ix+1, iy ) + Bz_tilde( ix, iy ) );
    num_flux_BE_y( 4, ix, iy ) =   0.;
    num_flux_BE_y( 5, ix, iy ) = + 0.5 * ( Bx_pls + Bx_mns ) - 0.5 * ( Ez_pls - Ez_mns );

  }}

  for( size_t i = 0; i < 6; i++  )
  {

    for( size_t ix = 0; ix < N[0]; ix++ ){
    for( size_t iy = 0; iy < N[1]; iy++ ){

      RHS( i, ix, iy ) = - ( num_flux_BE_x( i, ix+1, iy   ) - num_flux_BE_x( i, ix, iy ) ) * dx_inv;
                         - ( num_flux_BE_y( i, ix  , iy+1 ) - num_flux_BE_y( i, ix, iy ) ) * dx_inv;

    }}

  }

}

void simulation::RK_step( ArrayND<double>& E_, ArrayND<double>& B_, 
                          const ArrayND<double>& RHS_EB_1_, const ArrayND<double>& RHS_EB_2_, 
                          const double a_1, const double a_2 )
{

  for( size_t ix = 0; ix < N[0]; ix++ ){
  for( size_t iy = 0; iy < N[1]; iy++ ){

    size_t jx = ix+BD;
    size_t jy = iy+BD;

    B_(0, jx, jy) = B(0, jx, jy) + a_1 * dt * RHS_EB_1_( 0, ix, iy ) + a_2 * dt * RHS_EB_2_( 0, ix, iy );
    B_(1, jx, jy) = B(1, jx, jy) + a_1 * dt * RHS_EB_1_( 1, ix, iy ) + a_2 * dt * RHS_EB_2_( 1, ix, iy );
    B_(2, jx, jy) = B(2, jx, jy) + a_1 * dt * RHS_EB_1_( 2, ix, iy ) + a_2 * dt * RHS_EB_2_( 2, ix, iy );

    E_(0, jx, jy) = E(0, jx, jy) + a_1 * dt * RHS_EB_1_( 3, ix, iy ) + a_2 * dt * RHS_EB_2_( 3, ix, iy );
    E_(1, jx, jy) = E(1, jx, jy) + a_1 * dt * RHS_EB_1_( 4, ix, iy ) + a_2 * dt * RHS_EB_2_( 4, ix, iy );
    E_(2, jx, jy) = E(2, jx, jy) + a_1 * dt * RHS_EB_1_( 5, ix, iy ) + a_2 * dt * RHS_EB_2_( 5, ix, iy );

  }}

  set_ghost_cells(E_);
  set_ghost_cells(B_);

}

void simulation::step()
{

  get_dt();

  get_RHS_BE( RHS_BE_0, E  , B   );
  RK_step   ( E_1, B_1, RHS_BE_0, RHS_BE_1, 1. , 0.  );

  get_RHS_BE( RHS_BE_1, E_1, B_1 );
  RK_step   ( E  , B  , RHS_BE_0, RHS_BE_1, 0.5, 0.5 );

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

  const std::string file_name = "/home/fs1/mw/Reconnection/mikePhy/output_" + std::to_string(num_outputs) + ".vti";

  long N_bytes_scalar, N_bytes_vector;

  write_vti_header( file_name, N_bytes_scalar, N_bytes_vector );

  print_mpi_vector( file_name, N_bytes_vector, E );
  print_mpi_vector( file_name, N_bytes_vector, B );

  write_vti_footer( file_name );

  num_outputs += 1;

}

void simulation::print_mpi_vector( std::string file_name, long& N_bytes_vector, const ArrayND<double>& field )
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
  
  for( size_t ix = BD; ix < N_bd[0] - BD; ix++ ){
  for( size_t iy = BD; iy < N_bd[1] - BD; iy++ ){
    
    size_t id = (ix-BD) * N[1] + (iy-BD);
    
    float_array_vector[3*id+0] = float( field(0,ix,iy) );
    float_array_vector[3*id+1] = float( field(1,ix,iy) );
    float_array_vector[3*id+2] = float( field(2,ix,iy) ); 
  }}
  
  // write data
  MPI_File_set_view(mpi_file, mpi_eof, vti_float3, vti_subarray_vector, "native", MPI_INFO_NULL);
  MPI_File_write_all(mpi_file, float_array_vector, N[0]*N[1], vti_float3, MPI_STATUS_IGNORE);
  
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
  // long bin_size_scalar = N_bytes_scalar + sizeof(uint64_t);// 2nd term is the size of the the leading integer announcing the numbers n the data chunk
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
         << "\" Direction=\"0 1 0 1 0 0 0 0 1\">" << std::endl;
    
    os << "      <FieldData>" << std::endl;
    os << "        <DataArray type=\"Float32\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">" << std::endl;
    os << "        "<< float(time) << std::endl;
    os << "        </DataArray>" << std::endl;
    os << "      </FieldData>" << std::endl;
        
		os << "    <Piece Extent=\"" << extend_l[0] << " " << extend_r[0] << " " 
                                 << extend_l[1] << " " << extend_r[1] << " " 
                                 << "0" << " " << "1" << "\">" << std::endl;
    
    os << "      <CellData>" << std::endl;
    os << "        <DataArray type=\"Float32\" Name=\"E\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << offset << "\">" << std::endl;
    os << "        </DataArray>" << std::endl;
    offset += bin_size_vector;
    os << "        <DataArray type=\"Float32\" Name=\"B\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << offset << "\">" << std::endl;
    os << "        </DataArray>" << std::endl;
    offset += bin_size_vector;
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

  start_i[0] = BD; 
  start_i[1] = BD; 
  end_i  [0] = N_bd[0] - BD;
  end_i  [1] = N_bd[1] - BD;

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

  // create derived MPI_datatypes for vti output
  int size_total[3] = { static_cast<int>(N_tot)             ,static_cast<int>(N_tot             ),1 };
  int mpi_start[3]  = { static_cast<int>(mpi_coords[0]*N[0]),static_cast<int>(mpi_coords[1]*N[1]),0 };
  int mpi_size[3]   = { static_cast<int>(              N[0]),static_cast<int>(              N[1]),1 };

  MPI_Type_contiguous(3, MPI_FLOAT, &vti_float3);
  MPI_Type_commit(&vti_float3);

  MPI_Type_create_subarray(3, size_total, mpi_size, mpi_start, MPI_ORDER_C, MPI_FLOAT, &vti_subarray);
  MPI_Type_commit(&vti_subarray);
  
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