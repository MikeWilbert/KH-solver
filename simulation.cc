#include "simulation.h"

// Parallelen vti output und anschließend ghost cell exchange. Dann steht das framework


simulation::simulation( const size_t N_, const size_t BD_, const double cfl_ ) :
N_tot(N_), 
BD(BD_), 
cfl(cfl_) 
{

  init_mpi();

  E.resize({3, N_bd[0], N_bd[1]});
  B.resize({3, N_bd[0], N_bd[1]});

  setup();

  print_vti();

}

void simulation::setup()
{
  time = 0.;

  L = 2.;
  dx = L / N_tot;

  for( size_t ix = BD; ix < N_bd[0] - BD; ix++ ){
  for( size_t iy = BD; iy < N_bd[1] - BD; iy++ ){

    double x_val = ( ix - BD + 0.5 ) * dx + mpi_coords[0] * N_tot / mpi_dims[0] * dx;
    double y_val = ( iy - BD + 0.5 ) * dx + mpi_coords[1] * N_tot / mpi_dims[1] * dx;

    E(0,ix,iy) = mpi_rank;
    E(1,ix,iy) = mpi_coords[0];
    E(2,ix,iy) = mpi_coords[1];

    B(0,ix,iy) = sin( (2.*M_PI/L) * x_val );
    B(1,ix,iy) = cos( (2.*M_PI/L) * y_val );
    B(2,ix,iy) = mpi_coords[1];


  }}

  set_ghost_cells(E);
  set_ghost_cells(B);

  // testing: check ghost vales
  for( size_t iy = BD; iy < N_bd[1] - BD; iy++ ){

    E(0,BD  ,iy) = E(0,BD-1,iy);
    E(0,BD+1,iy) = E(0,BD-2,iy);
    E(0,N_bd[0]-3,iy) = E(0,N_bd[0]-1,iy);
    E(0,N_bd[0]-4,iy) = E(0,N_bd[0]-2,iy);

    B(0,BD  ,iy) = B(0,BD-1,iy);
    B(0,BD+1,iy) = B(0,BD-2,iy);
    B(0,N_bd[0]-3,iy) = B(0,N_bd[0]-1,iy);
    B(0,N_bd[0]-4,iy) = B(0,N_bd[0]-2,iy);

  }

}

void simulation::set_ghost_cells( ArrayND<double>& field )
{

  size_t num_fields = field.dim_size( 0 );

  std::vector<double> buffer( N_bd[0]*N_bd[1] );

  // exchange ghost cells for each component
  for( size_t i = 0; i < num_fields; i++ )
  {

    // fill buffer for field component
    for( size_t ix = 0; ix < N_bd[0]; ix++ ){
    for( size_t iy = 0; iy < N_bd[1]; iy++ ){

      size_t id = ix * N_bd[1] + iy;

      buffer[id] = field(i,ix,iy);

    }}

    // W -> E
    MPI_Sendrecv( buffer.data(), 1, mpi_slice_inner_W, mpi_neighbors[0], 123,
                  buffer.data(), 1, mpi_slice_outer_E, mpi_neighbors[1], 123, cart_comm, MPI_STATUS_IGNORE);

    // E -> W
    MPI_Sendrecv( buffer.data(), 1, mpi_slice_inner_E, mpi_neighbors[1], 123,
                  buffer.data(), 1, mpi_slice_outer_W, mpi_neighbors[0], 123, cart_comm, MPI_STATUS_IGNORE);

    // S -> N
    MPI_Sendrecv( buffer.data(), 1, mpi_slice_inner_S, mpi_neighbors[2], 123,
                  buffer.data(), 1, mpi_slice_outer_N, mpi_neighbors[3], 123, cart_comm, MPI_STATUS_IGNORE);

    // N -> S
    MPI_Sendrecv( buffer.data(), 1, mpi_slice_inner_N, mpi_neighbors[3], 123,
                  buffer.data(), 1, mpi_slice_outer_S, mpi_neighbors[2], 123, cart_comm, MPI_STATUS_IGNORE);

    // get updated field components from buffer
    for( size_t ix = 0; ix < N_bd[0]; ix++ ){
    for( size_t iy = 0; iy < N_bd[1]; iy++ ){

      size_t id = ix * N_bd[1] + iy;

      field(i,ix,iy) = buffer[id];

    }}

  }

}

void simulation::print_vti()
{

  const std::string file_name = "/home/fs1/mw/Reconnection/mikePhy/output.vti";

  long N_bytes_scalar, N_bytes_vector;

  write_vti_header( file_name, N_bytes_scalar, N_bytes_vector );

  print_mpi_vector( file_name, N_bytes_vector, E );
  print_mpi_vector( file_name, N_bytes_vector, B );

  write_vti_footer( file_name );

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