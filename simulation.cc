#include "simulation.h"

// Parallelen vti output und anschließend ghost cell exchange. Dann steht das framework


simulation::simulation( const size_t N_, const size_t BD_, const double cfl_ ) :
N_tot(N_), 
BD(BD_), 
cfl(cfl_) 
{

  init_mpi();

  N[0] = N_tot / mpi_dims[0];
  N[1] = N_tot / mpi_dims[1];

  N_bd[0] = N[0] + 2*BD;
  N_bd[1] = N[1] + 2*BD;

  // create derived MPI_datatypes for vti output
  int size_total[3] = { N_tot,N_tot,1 };
  int mpi_start[3]  = { mpi_coords[0]*N[0],mpi_coords[1]*N[1],0 };
  int mpi_size[3]   = {               N[0],              N[1],1 };

  MPI_Type_contiguous(3, MPI_FLOAT, &vti_float3);
  MPI_Type_commit(&vti_float3);

  MPI_Type_create_subarray(3, size_total, mpi_size, mpi_start, MPI_ORDER_C, MPI_FLOAT, &vti_subarray);
  MPI_Type_commit(&vti_subarray);
  
  MPI_Type_create_subarray(3, size_total,mpi_size, mpi_start, MPI_ORDER_C, vti_float3, &vti_subarray_vector);
  MPI_Type_commit(&vti_subarray_vector);

  float_array_vector.resize( {3,N[0],N[1]} );

  E.resize({3, N_bd[0], N_bd[1]});
  B.resize({3, N_bd[0], N_bd[1]});

  setup();

  print_vti();

}

void simulation::setup()
{

  L = 2.;
  dx = L / N_tot;
  time = 0.;

  for( size_t ix = BD; ix < N_bd[0] - BD; ix++ ){
  for( size_t iy = BD; iy < N_bd[1] - BD; iy++ ){

    double x_val = ( ix - BD + 0.5 ) * dx;
    double y_val = ( iy - BD + 0.5 ) * dx;

    E(0,ix,iy) = sin( L/(2.*M_PI) * x_val );
    E(1,ix,iy) = cos( L/(2.*M_PI) * x_val );
    E(2,ix,iy) = 1.;

    B(0,ix,iy) = sin( L/(2.*M_PI) * y_val );
    B(1,ix,iy) = cos( L/(2.*M_PI) * y_val );
    B(2,ix,iy) = 1.;

  }}

}

void simulation::print_vti()
{

  const std::string file_name = "/home/fs1/mw/Reconnection/mikePhy/output.vti";

  write_vti_header( file_name );



  write_vti_footer( file_name );

}

void simulation::print_mpi_vector(long& N_bytes_vector, const char* file_name)
{
  if(mpi_rank==0)
  {
    std::ofstream binary_os(file_name, std::ios::out | std::ios::app | std::ios::binary );
    binary_os.write(reinterpret_cast<const char*>(&N_bytes_vector),sizeof(uint64_t)); // size of following binary package
    binary_os.close();
  }MPI_Barrier(MPI_COMM_WORLD);
  
  // open file
  MPI_File mpi_file;
  MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_APPEND|MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_file);
  
  // offset to end of file
  MPI_Offset mpi_eof;
  MPI_File_get_position(mpi_file, &mpi_eof);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // data to float array
  // for(int id = 0; id < size_R_tot; id++)
  // {
    // float_array_vector[3*id+0] = float(field_X[id].real());
    // float_array_vector[3*id+1] = float(field_Y[id].real());
    // float_array_vector[3*id+2] = float(field_Z[id].real());
  // }
  
  // write data
  MPI_File_set_view(mpi_file, mpi_eof, vti_float3, vti_subarray_vector, "native", MPI_INFO_NULL);
  // MPI_File_write_all(mpi_file, float_array_vector, size_R_tot, vti_float3, MPI_STATUS_IGNORE);
  
  // close file
  MPI_File_close(&mpi_file);  
}

void simulation::write_vti_header( std::string file_name )
{

  std::ofstream os;
  
  long N_l = N_tot;
  long offset = 0;
	long N_tot = N_l*N_l;
	long N_bytes_scalar  =   N_tot * sizeof(float);
	long N_bytes_vector  = 2*N_tot * sizeof(float);
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
		int extend_r[2]  = {N_tot, N_tot};
		double origin[3] = {0.,0.,0.};
    
    os << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">" << std::endl;	
    os << "  <ImageData WholeExtent=\"" << extend_l[0] << " " << extend_r[0] << " " 
                                        << extend_l[1] << " " << extend_r[1] << " " 
                                        << "0" << " " << "0"
				 << "\" Origin=\""  << origin[0]  << " " << origin[1]  << " " << origin[2] 
				 << "\" Spacing=\"" << dx << " " << dx << " " << 1.
         << "\" Direction=\"0 0 1 0 1 0 1 0 0\">" << std::endl; // FORTRAN -> C order (no effect..)
    
    os << "      <FieldData>" << std::endl;
    os << "        <DataArray type=\"Float32\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">" << std::endl;
    os << "        "<< float(time) << std::endl;
    os << "        </DataArray>" << std::endl;
    os << "      </FieldData>" << std::endl;
        
		os << "    <Piece Extent=\"" << extend_l[0] << " " << extend_r[0] << " " 
                                 << extend_l[1] << " " << extend_r[1] << " " 
                                 << "0" << " " << "0" << "\">" << std::endl;
    
    os << "      <PointData>" << std::endl;
    os << "      </PointData>" << std::endl;
    os << "      <CellData>" << std::endl;
    os << "        <DataArray type=\"Float32\" Name=\"E\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << offset << "\">" << std::endl;
    os << "        </DataArray>" << std::endl;
    offset += bin_size_vector;
    os << "        <DataArray type=\"Float32\" Name=\"B\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << offset << "\">" << std::endl;
    os << "        </DataArray>" << std::endl;
    offset += bin_size_vector;
    os << "      </CellData>" << std::endl;
    os << "    </Piece>" << std::endl;
    os << "  </ImageData>" << std::endl;
    os << "  <AppendedData encoding=\"raw\">" << std::endl;
    os << "   _";
                                
    os.close();
  
  }MPI_Barrier(MPI_COMM_WORLD);

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
  }MPI_Barrier(MPI_COMM_WORLD);

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