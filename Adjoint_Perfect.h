/* Put the Trilinos and CFFC includes right here */
// C++ Reading/Writing
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include "../MPI/MPI.h"

//Trilinos
#include "AztecOO_config.h"
#include <Epetra_config.h>
#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include <stdexcept>
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_CrsGraph.h"
#include "AztecOO.h"
#include <Ifpack.h>
#include <Ifpack_ConfigDefs.h>
#include <Ifpack_AdditiveSchwarz.h>
#include <boost/date_time/posix_time/posix_time.hpp>


// CFFC
#include "../../bpkit/src/DenseMat.h" // DenseMat included
#include "../../bpkit/src/BlockMat.h" // BlockMat included
#include "../../bpkit/src/BILUK.h" // BILUK included
#include "../../bpkit/src/BRelax.h" // BRelax included
#include "../Math/Matrix.h"
#include "../Math/Vector2D.h"
// #include "../NewtonKrylovSchwarz/NKS_DTS.h"

#ifndef _OCTREE_INCLUDED
#include "../AMR/Octree.h"
#endif // _OCTREE_INCLUDED

#ifndef _GRID3D_HEXA_MULTIBLK_INCLUDED
#include "../Grid/Grid3DHexaMultiBlock.h"
#endif // _GRID3D_HEXA_MULTIBLK_INCLUDED

/* Include Morton re-ordering header file. */

#ifndef _MORTON_ORDERING_INCLUDED
#include "MortonOrdering.h"
#endif // _MORTON_ORDERING_INCLUDED


//*
enum threeDStencils { FIRST_ORDER_stencil = 7,
		      SECOND_ORDER_stencil = 27 };                      //WITH HYBRID  JUST USING 7 


enum locations {  stencil_CENTER = 0,		  // enum2 done so as to compare files
                  stencil_NORTH = 2,
		  stencil_SOUTH = 1,
		  stencil_EAST  = 4,
		  stencil_WEST  = 3,
		  stencil_TOP   = 6,
		  stencil_BOTTOM = 5 }; 


// ==========================================================================
// GLOBAL VARIABLES


// Declare the Epetra variables here and ensure they remain global
//  Variables for Epetra needed later:
	  int Ni=0, Nj=0, Nk=0, blocksize=0, global_numBlk=0;
	  int Jacobian_stencil_size = 7;
	  int NCi=0, NCj=0, NCk=0, Nghost=0;
	  int JCl_overlap = 0, JCu_overlap = 0, ICu_overlap =0, ICl_overlap = 0, KCl_overlap = 0, KCu_overlap = 0;
	  int ICl=0, JCl=0, KCl=0, ICu=0, JCu=0, KCu=0;
	  int N=0, M=0, O=0, NumGlobalElements=0,  MyElements=0,  indexBase=0,  n=0, availProcs=0, MyPID =0;
	  Epetra_MpiComm *Comm = NULL;
	  Epetra_Map *Map2 = NULL;
	  Epetra_Vector *bb = NULL; 
	
// 	  Epetra_Vector *x=NULL;
// 	  Epetra_Vector *top=NULL;
	  Epetra_CrsMatrix *AA = NULL;	
	  DenseMatrix *Jacobian_Data = NULL; 
	  DenseMatrix *TempMat = NULL;
	  int ErrA;
	  int numProcs=0;
	  int prevVal;
	  int numProcsUsed = 0;
	  int LocalBlocksinProc = 0;
	  int PrevGlobBlock = 0;
	  int prevBlock = 0;
	  int BlockinProc = 0;
	  
	  int WriteMatrixtoFile = 0;
	  int PrintSparse = 0;
	  int WriteVector = 0;
	  
	  int Transpos = 1; // set to 1 to get transpose of A
	  
	  // The type of global indices.  You could just set this to int,
	    // but we want the example to work for Epetra64 as well.
	  #ifdef EPETRA_NO_32BIT_GLOBAL_INDICES
	    // Epetra was compiled only with 64-bit global index support, so use
	    // 64-bit global indices.
	    typedef long long global_ordinal_type;
	  #else
	    // Epetra was compiled with 32-bit global index support.  If
	    // EPETRA_NO_64BIT_GLOBAL_INDICES is defined, it does not also
	    // support 64-bit indices.
	    typedef int global_ordinal_type;
	  #endif // EPETRA_NO_32BIT_GLOBAL_INDICES
	  global_ordinal_type* MyGlobalElements = NULL;  
	  
	  std::vector <int> ProcList;
	  std::vector <int> indx;
	  std::vector<int> BlockList;
	  std::vector <int> Cummul_Blocks;
	  std::vector <std::vector < std::vector <int> > > PosBlock; 
	  

/***************************************************************************
 * Newer Get Block Index----------------------------------------------------------
*******************************************************************************/

template <typename SOLN_pSTATE, typename SOLN_cSTATE>    
void NewGet_Block_Index(const int &cell_index_i, const int &cell_index_j, const int &cell_index_k,int *Block_index_i, int *Block_index_j, 
		     int BlkN, int BlkS, int BlkE, int BlkW, int BlkT, int BlkB, int nBlkN, int nBlkS, int nBlkE, int nBlkW, int nBlkT, int nBlkB,
		     int cpuN, int cpuS, int cpuE, int cpuW, int cpuT, int cpuB, int ii, int jj, int kk		   
		    )
{   
 /*****************************************************************************
 *  Establish the index of neighbours of cells via the stencil                        *
 *****************************************************************************/   
 
  int iNeigh, jNeigh, kNeigh;
  int CellID, neighCellID;
  int BlkUsed, cpudel, cpuUsed;
 
    CellID =  cell_index_k*((Ni)*(Nj)) + cell_index_j*(Ni)+cell_index_i;
    
    for( int i=0; i<Jacobian_stencil_size; i++)  
    Block_index_i[i] = prevBlock * (N*M*O) +  CellID; 

    //I index 
    if( Jacobian_stencil_size == FIRST_ORDER_stencil){        
      /*! Determine 1st order Block_Indicies for Cell (i,j)   
      *                    
      *            ---    6
      *           | 2 | / 
      *        --- --- ---
      *       | 3 | 0 | 4 |                
      *        --- --- ---
      *         / | 1 |
      *       5    ---
      */ 
   
      
      Block_index_j[stencil_CENTER] = prevBlock * (N*M*O) +  CellID;    // 2  
    
      for (int a=1; a<7; a++){
	Block_index_j[a]= -1;
      }
      
      // WEST========================================================
      if (nBlkW !=0 && cell_index_i == ICl-Nghost) {   // i.e. if i = 0 ... change this to check for Blocks on the west side	  
	  iNeigh = N-1; 
	  jNeigh = jj;
	  kNeigh = kk;
	  cpuUsed = cpuW;
	  BlkUsed = BlkW;
	  neighCellID = iNeigh + jNeigh*N + kNeigh * N*M;
	  
	  if (cpuUsed != 0){
	    Block_index_j[stencil_WEST] = (Cummul_Blocks[cpuUsed - 1] + BlkUsed)*(N*M*O) + neighCellID;	  
	  }
	  else{
	    Block_index_j[stencil_WEST] = (0 + BlkUsed)*(N*M*O) + neighCellID;	  
	  }
// 	  cpudel = (Cummul_Blocks[cpuUsed] + BlkUsed - prevBlock) * N*M*O;
// 	  neighcol = cpudel + (prevBlock*N*M*O) +  neighCellID;
// 	  Block_index_j[stencil_WEST]= -1; // replace this
      }
      if (cell_index_i != ICl-Nghost) {	
	  Block_index_j[stencil_WEST] = Block_index_j[stencil_CENTER] -1;
      }
// EAST========================================================
      if (nBlkE !=0 && cell_index_i == ICu -Nghost){	
// 	  Block_index_j[stencil_EAST]= -1;  
	  iNeigh = 0; 
	  jNeigh = jj;
	  kNeigh = kk;
	  cpuUsed = cpuE;
	  BlkUsed = BlkE;
	  neighCellID = iNeigh + jNeigh*N + kNeigh * N*M;
	  if (cpuUsed != 0){
	    Block_index_j[stencil_EAST] = (Cummul_Blocks[cpuUsed - 1] + BlkUsed)*(N*M*O) + neighCellID;	  
	  }
	  else{
	    Block_index_j[stencil_EAST] = (0 + BlkUsed)*(N*M*O) + neighCellID;	  
	  }
      }
      if (cell_index_i != ICu -Nghost){	
	  Block_index_j[stencil_EAST] = Block_index_j[stencil_CENTER] +1;
      }
// SOUTH=========================================================
      if (nBlkS !=0 && cell_index_j == JCl-Nghost) {
// 	  Block_index_j[stencil_SOUTH]= -1;
	  iNeigh = ii; 
	  jNeigh = M-1;
	  kNeigh = kk;
	  BlkUsed = BlkS;
	  cpuUsed = cpuS;
	  neighCellID = iNeigh + jNeigh*N + kNeigh * N*M;
	  if (cpuUsed != 0){
	    Block_index_j[stencil_SOUTH] = (Cummul_Blocks[cpuUsed - 1] + BlkUsed)*(N*M*O) + neighCellID;	  
	  }
	  else{
	    Block_index_j[stencil_SOUTH] = (0 + BlkUsed)*(N*M*O) + neighCellID;	  
	  }
      }
      if (cell_index_j != JCl-Nghost) {	
	  Block_index_j[stencil_SOUTH] = Block_index_j[stencil_CENTER] - (Ni);
      }   
// NORTH=========================================================
      if (nBlkN !=0 && cell_index_j == JCu-Nghost) {	
// 	  Block_index_j[stencil_NORTH]= -1;
	  iNeigh = ii; 
	  jNeigh = 0; 
	  kNeigh = kk;
	  BlkUsed = BlkN;
	  cpuUsed = cpuN;
	  neighCellID = iNeigh + jNeigh*N + kNeigh * N*M;
	  if (cpuUsed != 0){
	    Block_index_j[stencil_NORTH] = (Cummul_Blocks[cpuUsed - 1] + BlkUsed)*(N*M*O) + neighCellID;	  
	  }
	  else{
	    Block_index_j[stencil_NORTH] = (0 + BlkUsed)*(N*M*O) + neighCellID;	  
	  }
      }
      if (cell_index_j != JCu-Nghost) {	
	  Block_index_j[stencil_NORTH] = Block_index_j[stencil_CENTER] + (Ni);
      }
        
 // BOTTOM ==============================================================      
 
      if (nBlkB !=0 && cell_index_k == KCl-Nghost) {	
// 	  Block_index_j[stencil_BOTTOM]= -1;
	  iNeigh = ii; 
	  jNeigh = jj;
	  kNeigh = O-1;
	  BlkUsed = BlkB;
	  cpuUsed = cpuB;
	  neighCellID = iNeigh + jNeigh*N + kNeigh * N*M;
	  if (cpuUsed != 0){
	    Block_index_j[stencil_BOTTOM] = (Cummul_Blocks[cpuUsed - 1] + BlkUsed)*(N*M*O) + neighCellID;	  
	  }
	  else{
	    Block_index_j[stencil_BOTTOM] = (0 + BlkUsed)*(N*M*O) + neighCellID;	  
	  }      
      }
      if (cell_index_k != KCl-Nghost) {	
	  Block_index_j[stencil_BOTTOM] = Block_index_j[stencil_CENTER] - (Ni*Nj);
      }

 // TOP =================================================================
      if (nBlkT !=0 && cell_index_k == KCu-Nghost) {	
// 	  Block_index_j[stencil_TOP]= -1;
	  iNeigh = ii; 
	  jNeigh = jj;
	  kNeigh = 0;
	  BlkUsed = BlkT;
	  cpuUsed = cpuT;
	  neighCellID = iNeigh + jNeigh*N + kNeigh * N*M;
	  if (cpuUsed != 0){
	    Block_index_j[stencil_TOP] = (Cummul_Blocks[cpuUsed - 1] + BlkUsed)*(N*M*O) + neighCellID;	  
	  }
	  else{
	    Block_index_j[stencil_TOP] = (0 + BlkUsed)*(N*M*O) + neighCellID;	  
	  }
      }     
      if (cell_index_k != KCu-Nghost) {	
	  Block_index_j[stencil_TOP] = Block_index_j[stencil_CENTER] + (Ni*Nj);
	
      }
    }  
    
    
    else if ( Jacobian_stencil_size == SECOND_ORDER_stencil) {
      cerr<<" Not using SECOND_ORDER_stencil in 3D  "; exit(1);
    }  

}


template <typename SOLN_pSTATE, typename SOLN_cSTATE> 
DenseMatrix Rotation_Matrix_3D(const Vector3D &norm_dir, int A_matrix) 
{
/*****************************************************************************
 *  Rotation and restoration as we move to different faces                              *
 *****************************************************************************/      	
  DenseMatrix mat(blocksize,blocksize);   //TEMP var
  mat.identity();
  
  double cos_psi, sin_psi, cos_phi, sin_phi, cos_theta, sin_theta;
  cos_phi = ONE;
  sin_phi = ZERO;
  if (fabs(fabs(norm_dir.x)-ONE) < TOLER) {
    cos_psi = norm_dir.x/fabs(norm_dir.x);
    sin_psi = ZERO;
    cos_theta = ONE;
    sin_theta = ZERO;
  } else {
    cos_psi = norm_dir.x;
    sin_psi = sqrt(norm_dir.y*norm_dir.y + norm_dir.z*norm_dir.z);
    cos_theta = norm_dir.y/sqrt(norm_dir.y*norm_dir.y + norm_dir.z*norm_dir.z);
    sin_theta = norm_dir.z/sqrt(norm_dir.y*norm_dir.y + norm_dir.z*norm_dir.z);
  } 
  
  if (A_matrix) {             
    // Rotation Matrix, A                 
    mat(1,1) = (cos_psi*cos_phi-cos_theta*sin_phi*sin_psi);
    mat(1,2) = (cos_psi*sin_phi+cos_theta*cos_phi*sin_psi);
    mat(1,3) = (sin_psi*sin_theta);
    mat(2,1) = (-sin_psi*cos_phi-cos_theta*sin_phi*cos_psi);
    mat(2,2) = (-sin_psi*sin_phi+cos_theta*cos_phi*cos_psi);
    mat(2,3) = (cos_psi*sin_theta);
    mat(3,1) = (sin_theta*sin_phi);
    mat(3,2) = (-sin_theta*cos_phi);
    mat(3,3) = (cos_theta);

    //Inverse
  }else {
    mat(1,1) = (cos_psi*cos_phi-cos_theta*sin_phi*sin_psi);
    mat(1,2) = (-sin_psi*cos_phi-cos_theta*sin_phi*cos_psi);
    mat(1,3) = (sin_theta*sin_phi);
    mat(2,1) = (cos_psi*sin_phi+cos_theta*cos_phi*sin_psi);
    mat(2,2) = (-sin_psi*sin_phi+cos_theta*cos_phi*cos_psi);
    mat(2,3) = (-sin_theta*cos_phi);
    mat(3,1) = (sin_theta*sin_psi);
    mat(3,2) = (sin_theta*cos_psi);
    mat(3,3) = (cos_theta);
  } 

  return mat;

} /* End of Rotation_Matrix. */
     


template <typename SOLN_pSTATE, typename SOLN_cSTATE> 
void dFIdW_Inviscid_ROE(DenseMatrix& dRdW, 
		   const SOLN_pSTATE &Wr, 
		   const SOLN_pSTATE &Wl,
		   const Vector3D &nface,
		   const double &Aface){//,
/*****************************************************************************
 *  Evaluate Rotation and Restoration matrix, get wavespeeds, and dR/dW                                      *
 *****************************************************************************/   		  
  //int blocksize = dRdW.get_n(); 
  static DenseMatrix dFidW(blocksize, blocksize,ZERO);
  dFidW.zero();

  DenseMatrix A(Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface, 1));			          //  errors here
  DenseMatrix AI(Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface, 0));
  
  SOLN_pSTATE Left(Wl.Rotate(nface));
  SOLN_pSTATE Right(Wr.Rotate(nface));
   
  /********** Analytical ************************************/  
  //Determine Roe Averaged State

  SOLN_pSTATE Wa(Wl.RoeAverage(Left,Right));    
    
  Left.dFxdW(dFidW);    // Leave it as left, ie. i,j,k
  dFidW *= HALF; 

  // Determine Wave Speeds
  SOLN_pSTATE wavespeeds( Wa.lambda_abs( Wa.lambda_x(),
					 Left.lambda_x(),      // Order is irrelevant
					 Right.lambda_x()) );         
    
  //Loop through each wavespeed and each element of Jacobian(i,j)        
  for (int i=1; i <= blocksize; i++) {		   
    for(int irow =0; irow< blocksize; irow++){
      for(int jcol =0; jcol< blocksize; jcol++){         
	dFidW(irow, jcol) += HALF*wavespeeds[i]*Wa.lp_x(i)[jcol+1]*Wa.rc_x(i)[irow+1];  // Don't change this
      }
    }
  } 
  /*********************************************************/
  
  //Rotate back 
  dRdW += Aface*AI*dFidW*A;    // Don't change
} 


//Inviscid Roe
template <typename SOLN_pSTATE, typename SOLN_cSTATE> 
void Preconditioner_dFIdU_Roe(DenseMatrix &_dFIdU, 
			      int i, int j, int k,  int Orient,  Hexa_Block<SOLN_pSTATE, SOLN_cSTATE> & MyBlock){//,
/**************************************************************************************************************************************
 *  Determine LHS and RHS, Normal, Area, and call Roe solver, evaluate dF/dU, our Flux Jacobian                                      *
 *************************************************************************************************************************************/     
  int Ri(i), Rj(j), Rk(k);   // copying values of (i,j,k) to (Ri,Rj,Rk)
  Vector3D nface;
  double Aface;
  //std::cout<<"Now inside the Preconditioner_dFIdU_Roe"<<endl;  
  switch(Orient){
      case stencil_SOUTH:
	Rj = j - 1;          // Leave as is  
	nface = MyBlock.Grid.nfaceS(i, j, k); 
	Aface = MyBlock.Grid.AfaceS(i, j, k);
	break;
      case stencil_NORTH:
	Rj = j + 1;
	nface = MyBlock.Grid.nfaceN(i, j, k);   // i.e the North face of the left cell which is: (i,j,k)
	Aface = MyBlock.Grid.AfaceN(i, j, k);
	break;
      case  stencil_WEST: 
	Ri = i - 1;
	nface = MyBlock.Grid.nfaceW(i, j, k);     
	Aface = MyBlock.Grid.AfaceW(i, j, k);
	break;
      case  stencil_EAST:     
	Ri = i + 1;
	nface = MyBlock.Grid.nfaceE(i, j, k);
	Aface = MyBlock.Grid.AfaceE(i, j, k);
	break;
      case  stencil_BOTTOM: 
	Rk = k - 1;
	nface = MyBlock.Grid.nfaceBot(i, j, k);
	Aface = MyBlock.Grid.AfaceBot(i, j, k);
	break;
      case stencil_TOP: 
	Rk = k + 1;
	nface = MyBlock.Grid.nfaceTop(i, j, k);
	Aface = MyBlock.Grid.AfaceTop(i, j, k);
	break;
  }
  
  static DenseMatrix dFI_dW(blocksize,blocksize,ZERO);
  dFI_dW.zero();
      
      dFIdW_Inviscid_ROE<SOLN_pSTATE,SOLN_cSTATE>(dFI_dW,    // Leave as is
			MyBlock.W[Ri][Rj][Rk],                         //Wr The Right state comes first
			MyBlock.W[i][j][k], 			  	//Wl The Left state comes after Right 
			nface, Aface);

  static DenseMatrix dWdU(blocksize,blocksize,ZERO);  
  dWdU.zero();
  MyBlock.W[i][j][k].dWdU(dWdU);     // Leave as is
  


  _dFIdU += dFI_dW*dWdU;	     // Leave as is	

}
 
template <typename SOLN_pSTATE, typename SOLN_cSTATE> 

void Initialize_Jacobian(DenseMatrix &JDat){ 
 /*****************************************************************************
 *  Jacobian Matrix used to store ones                                      *
 *****************************************************************************/   
   
//std::cout<<"a is "<< a <<endl;
    for (int i=0; i<blocksize; i++){
	for (int j=0; j<blocksize; j++){

	  JDat(i,j) = 1.0;  
    //     std::cout<<"Jacobian_Data("<<i<<","<<j<<") is "<<Jacobian_Data(i,j)<<endl;
	}
    }
}


template <typename SOLN_pSTATE, typename SOLN_cSTATE> 
void Initialize_TempM(DenseMatrix &TMat){ 
/*****************************************************************************
 *  Temporary Matrix used to store zeros                                       *
 *****************************************************************************/    
    for (int i=0; i<blocksize; i++){
	for (int j=0; j<blocksize; j++){
	  TMat(i,j) = 0;  
	}
    }
}


template <typename SOLN_pSTATE, typename SOLN_cSTATE> 
void FillMatrix(int row, int col, DenseMatrix JBlock) {
 /*****************************************************************************
 *  Epetra Fill Matrix                                       *
 *****************************************************************************/   
  double temp_val, temp_val_T;
  int temp_col;
 
  for (int icount=0; icount<blocksize; icount++){  // icount is  for the 5x5 matrix
    for (int jcount=0; jcount<blocksize; jcount++){
	
	temp_val = JBlock(icount,jcount);
	temp_val_T = JBlock(jcount, icount);
	temp_col = ((col*blocksize)+jcount);
	
	if (Transpos = 1){
	  ErrA = AA->InsertGlobalValues( (row*blocksize)+icount, 1, &temp_val_T, &temp_col);
	}
	else{
	  ErrA = AA->InsertGlobalValues( (row*blocksize)+icount, 1, &temp_val, &temp_col);
	}
		
	if (ErrA !=0){
	  std::cout<<"Epetra FillMatrix: I have an error for (row,col) = ("<<row<<","<<col<<") located on line "<<(row*blocksize)+icount<<" of value= "<<ErrA<<std::endl;
	  exit(123);
	}
    }
  }
  
}

template <typename SOLN_pSTATE, typename SOLN_cSTATE> 
DenseMat DenseMatrix_to_DenseMat(const DenseMatrix &B) {
/*****************************************************************************
 *  Not needed for now                                        *
 *****************************************************************************/    
  DenseMat A(B.dim(0),B.dim(1));
  for (int i=0; i<B.dim(0); i++) {
    for( int j=0; j<B.dim(1); j++) {
      A(i,j) = B(i,j); 
    }
  }
  return A;
} 

template <typename SOLN_pSTATE, typename SOLN_cSTATE> 
void First_Order_Inviscid_Jacobian_Roe(const int &cell_index_i,
				       const int &cell_index_j,
				       const int &cell_index_k, 
				       DenseMatrix* Jacobian, 
				       Hexa_Block<SOLN_pSTATE, SOLN_cSTATE> &MyBlock){//,
				       //Input_Parameters<SOLN_pSTATE, SOLN_cSTATE> &Input){  
/*****************************************************************************
 *  Solve Roe Flux given Right and Left States                                       *
 *****************************************************************************/  					 
    Preconditioner_dFIdU_Roe(Jacobian[stencil_NORTH],cell_index_i,cell_index_j,cell_index_k,stencil_NORTH, MyBlock);//, Input); 
    Preconditioner_dFIdU_Roe(Jacobian[stencil_SOUTH],cell_index_i,cell_index_j,cell_index_k,stencil_SOUTH, MyBlock);//, Input);
    Preconditioner_dFIdU_Roe(Jacobian[stencil_EAST],cell_index_i,cell_index_j,cell_index_k,stencil_EAST, MyBlock);//, Input); 
    Preconditioner_dFIdU_Roe(Jacobian[stencil_WEST],cell_index_i,cell_index_j,cell_index_k,stencil_WEST, MyBlock);//, Input);        
    Preconditioner_dFIdU_Roe(Jacobian[stencil_TOP],cell_index_i,cell_index_j,cell_index_k,stencil_TOP, MyBlock);//, Input); 
    Preconditioner_dFIdU_Roe(Jacobian[stencil_BOTTOM],cell_index_i,cell_index_j,cell_index_k,stencil_BOTTOM, MyBlock);//, Input); 
    
  //Center calculated from neighbours
  //! Using the fact that dF/dU(right) = - dF/dU(left)  
  
 // below works with option enum2 at the beginning of file - Don't change
  Jacobian[stencil_CENTER] -= (Jacobian[stencil_NORTH] + Jacobian[stencil_SOUTH] + Jacobian[stencil_EAST] + Jacobian[stencil_WEST] + Jacobian[stencil_BOTTOM] + Jacobian[stencil_TOP])/MyBlock.Grid.Cell[cell_index_i][cell_index_j][cell_index_k].V;
  Jacobian[stencil_NORTH]  = Jacobian[stencil_NORTH]/MyBlock.Grid.Cell[cell_index_i][cell_index_j+1][cell_index_k].V;
  Jacobian[stencil_SOUTH]  = Jacobian[stencil_SOUTH]/MyBlock.Grid.Cell[cell_index_i][cell_index_j-1][cell_index_k].V;
  Jacobian[stencil_EAST]   = Jacobian[stencil_EAST]/MyBlock.Grid.Cell[cell_index_i+1][cell_index_j][cell_index_k].V; 
  Jacobian[stencil_WEST]   = Jacobian[stencil_WEST]/MyBlock.Grid.Cell[cell_index_i-1][cell_index_j][cell_index_k].V;
  Jacobian[stencil_TOP]    = Jacobian[stencil_TOP]/MyBlock.Grid.Cell[cell_index_i][cell_index_j][cell_index_k+1].V;
  Jacobian[stencil_BOTTOM] = Jacobian[stencil_BOTTOM]/MyBlock.Grid.Cell[cell_index_i][cell_index_j][cell_index_k-1].V;

}

/*!***********************************************************
 *  dFdU needs to be Provided By _Quad_Block Specialization  *
 *  otherwise the following errors will be shown.            *
 *************************************************************/
template <typename SOLN_pSTATE, typename SOLN_cSTATE> 
void Preconditioner_dFIdU(DenseMatrix &_dFdU, SOLN_pSTATE W) {
     W.dFxdU(_dFdU);
}


/*****************************************************************************
 *  Calculate First Order Local Jacobian Block(s) Coresponding to Cell(i,j)  *
 *  using HLLE                                                               *
 *****************************************************************************/ 
template <typename SOLN_pSTATE, typename SOLN_cSTATE> 
void First_Order_Inviscid_Jacobian_HLLE(const int &cell_index_i,
					const int &cell_index_j,
					const int &cell_index_k, 
					DenseMatrix* Jacobian, 
					Hexa_Block<SOLN_pSTATE, SOLN_cSTATE> & MyBlock){  
   
  //! Calculate normal vectors -> in Vector3D format. 
  Vector3D nface_N = MyBlock.Grid.nfaceN(cell_index_i,cell_index_j+1,cell_index_k);
  Vector3D nface_S = MyBlock.Grid.nfaceS(cell_index_i,cell_index_j-1,cell_index_k);      
  Vector3D nface_E = MyBlock.Grid.nfaceE(cell_index_i+1,cell_index_j,cell_index_k);
  Vector3D nface_W = MyBlock.Grid.nfaceW(cell_index_i-1,cell_index_j,cell_index_k);
  Vector3D nface_Top = MyBlock.Grid.nfaceTop(cell_index_i,cell_index_j,cell_index_k+1);
  Vector3D nface_Bot = MyBlock.Grid.nfaceBot(cell_index_i,cell_index_j,cell_index_k-1);
  
  

  //! Calculate wavespeeds using solutions in the rotated frame -> in Vector2D format.  (lambda+, lambda-)
  Vector2D lambdas_N = MyBlock.W[cell_index_i][cell_index_j][cell_index_k].HLLE_wavespeeds(
				       MyBlock.W[cell_index_i][cell_index_j+1][cell_index_k], 
				       MyBlock.W[cell_index_i][cell_index_j][cell_index_k], nface_N);
  Vector2D lambdas_S = MyBlock.W[cell_index_i][cell_index_j][cell_index_k].HLLE_wavespeeds(
				       MyBlock.W[cell_index_i][cell_index_j-1][cell_index_k], 
				       MyBlock.W[cell_index_i][cell_index_j][cell_index_k], nface_S);  
  Vector2D lambdas_E = MyBlock.W[cell_index_i][cell_index_j][cell_index_k].HLLE_wavespeeds(
                                       MyBlock.W[cell_index_i+1][cell_index_j][cell_index_k], 
				       MyBlock.W[cell_index_i][cell_index_j][cell_index_k], nface_E);
  Vector2D lambdas_W = MyBlock.W[cell_index_i][cell_index_j][cell_index_k].HLLE_wavespeeds(
				       MyBlock.W[cell_index_i-1][cell_index_j][cell_index_k], 
				       MyBlock.W[cell_index_i][cell_index_j][cell_index_k], nface_W);
  Vector2D lambdas_Top = MyBlock.W[cell_index_i][cell_index_j][cell_index_k].HLLE_wavespeeds(
					 MyBlock.W[cell_index_i][cell_index_j][cell_index_k+1], 
					 MyBlock.W[cell_index_i][cell_index_j][cell_index_k], nface_Top);
  Vector2D lambdas_Bot = MyBlock.W[cell_index_i][cell_index_j][cell_index_k].HLLE_wavespeeds(
					 MyBlock.W[cell_index_i][cell_index_j][cell_index_k-1], 
					 MyBlock.W[cell_index_i][cell_index_j][cell_index_k], nface_Bot);

  //! Calculate constants gamma and beta -> scalar values. 
  double gamma_N = (lambdas_N.x*lambdas_N.y)/(lambdas_N.y-lambdas_N.x);
  double beta_N  = - lambdas_N.x/(lambdas_N.y-lambdas_N.x);
  double gamma_S = (lambdas_S.x*lambdas_S.y)/(lambdas_S.y-lambdas_S.x);
  double beta_S  = - lambdas_S.x/(lambdas_S.y-lambdas_S.x);
  double gamma_E = (lambdas_E.x*lambdas_E.y)/(lambdas_E.y-lambdas_E.x);
  double beta_E  = - lambdas_E.x/(lambdas_E.y-lambdas_E.x);
  double gamma_W = (lambdas_W.x*lambdas_W.y)/(lambdas_W.y-lambdas_W.x);
  double beta_W  = - lambdas_W.x/(lambdas_W.y-lambdas_W.x);  
  double gamma_Top = (lambdas_Top.x*lambdas_Top.y)/(lambdas_Top.y-lambdas_Top.x);
  double beta_Top  = - lambdas_Top.x/(lambdas_Top.y-lambdas_Top.x);
  double gamma_Bot = (lambdas_Bot.x*lambdas_Bot.y)/(lambdas_Bot.y-lambdas_Bot.x);
  double beta_Bot  = - lambdas_Bot.x/(lambdas_Bot.y-lambdas_Bot.x);

  //! Obtain rotation matrices with normal vector -> matrices in DenseMatrix format. 
  DenseMatrix A_N( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_N, 1) );
  DenseMatrix AI_N( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_N, 0));
  DenseMatrix A_S( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_S, 1) );
  DenseMatrix AI_S( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_S, 0) );   //LOTS OF TEMPORARIES !!!!!!!!!! MAKE STATIC???
  DenseMatrix A_E( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_E, 1) );
  DenseMatrix AI_E( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_E, 0) );
  DenseMatrix A_W( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_W, 1) );
  DenseMatrix AI_W( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_W, 0) );
  DenseMatrix A_Top( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_Top, 1) );
  DenseMatrix AI_Top( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_Top, 0) );
  DenseMatrix A_Bot( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_Bot, 1) );
  DenseMatrix AI_Bot( Rotation_Matrix_3D<SOLN_pSTATE, SOLN_cSTATE>(nface_Bot, 0) );

  //! Calculate dFdU using solutions in the rotated frame -> matrix in DenseMatrix format. 
  DenseMatrix dFdU_N(blocksize,blocksize,ZERO); 
  DenseMatrix dFdU_S(blocksize,blocksize,ZERO); 
  DenseMatrix dFdU_E(blocksize,blocksize,ZERO); 
  DenseMatrix dFdU_W(blocksize,blocksize,ZERO); 
  DenseMatrix dFdU_Top(blocksize,blocksize,ZERO); 
  DenseMatrix dFdU_Bot(blocksize,blocksize,ZERO); 

  
  //Solution Rotate provided in pState 
  Preconditioner_dFIdU<SOLN_pSTATE,SOLN_cSTATE>( dFdU_N, MyBlock.W[cell_index_i][cell_index_j][cell_index_k].Rotate(nface_N));   
  Preconditioner_dFIdU<SOLN_pSTATE,SOLN_cSTATE>( dFdU_S, MyBlock.W[cell_index_i][cell_index_j][cell_index_k].Rotate(nface_S));
  Preconditioner_dFIdU<SOLN_pSTATE,SOLN_cSTATE>( dFdU_E, MyBlock.W[cell_index_i][cell_index_j][cell_index_k].Rotate(nface_E));
  Preconditioner_dFIdU<SOLN_pSTATE,SOLN_cSTATE>( dFdU_W, MyBlock.W[cell_index_i][cell_index_j][cell_index_k].Rotate(nface_W));
  Preconditioner_dFIdU<SOLN_pSTATE,SOLN_cSTATE>( dFdU_Top, MyBlock.W[cell_index_i][cell_index_j][cell_index_k].Rotate(nface_Top));
  Preconditioner_dFIdU<SOLN_pSTATE,SOLN_cSTATE>( dFdU_Bot, MyBlock.W[cell_index_i][cell_index_j][cell_index_k].Rotate(nface_Bot));
//   
  DenseMatrix II(blocksize,blocksize);  II.identity();     

  //! Calculate Jacobian matrix -> blocksizexblocksize matrix in DenseMatrix format
  //North
  Jacobian[stencil_NORTH] = (MyBlock.Grid.AfaceN(cell_index_i,cell_index_j+1,cell_index_k) 
		 * AI_N * (beta_N * dFdU_N + gamma_N * II) * A_N); 

  //South
  Jacobian[stencil_SOUTH] = (MyBlock.Grid.AfaceS(cell_index_i,cell_index_j-1,cell_index_k) 
		 * AI_S * (beta_S * dFdU_S + gamma_S * II) * A_S);

  //East
  Jacobian[stencil_EAST] = (MyBlock.Grid.AfaceE(cell_index_i+1,cell_index_j,cell_index_k) 
		 * AI_E * (beta_E * dFdU_E + gamma_E * II) * A_E);

  //West
  Jacobian[stencil_WEST] = (MyBlock.Grid.AfaceW(cell_index_i-1,cell_index_j,cell_index_k) 
		 * AI_W * (beta_W * dFdU_W + gamma_W * II) * A_W);
  
  //Top
  Jacobian[stencil_TOP] = (MyBlock.Grid.AfaceTop(cell_index_i,cell_index_j,cell_index_k+1) 
			   * AI_Top * (beta_Top * dFdU_Top + gamma_Top * II) * A_Top);

  //Bottom
  Jacobian[stencil_BOTTOM] = (MyBlock.Grid.AfaceBot(cell_index_i,cell_index_j,cell_index_k-1) 
			      * AI_Bot * (beta_Bot * dFdU_Bot + gamma_Bot * II) * A_Bot);


  //Center calculated from neighbours
  //! Using the fact that dF/dU(right) = - dF/dU(left) 
  
  Jacobian[stencil_CENTER] -= (Jacobian[stencil_NORTH] + Jacobian[stencil_SOUTH] + Jacobian[stencil_EAST] + Jacobian[stencil_WEST] + Jacobian[stencil_BOTTOM] + Jacobian[stencil_TOP])/MyBlock.Grid.Cell[cell_index_i][cell_index_j][cell_index_k].V;

  Jacobian[stencil_NORTH]  = Jacobian[stencil_NORTH]/MyBlock.Grid.Cell[cell_index_i][cell_index_j+1][cell_index_k].V;
  Jacobian[stencil_SOUTH]  = Jacobian[stencil_SOUTH]/MyBlock.Grid.Cell[cell_index_i][cell_index_j-1][cell_index_k].V;
  Jacobian[stencil_EAST]   = Jacobian[stencil_EAST]/MyBlock.Grid.Cell[cell_index_i+1][cell_index_j][cell_index_k].V;
  Jacobian[stencil_WEST]   = Jacobian[stencil_WEST]/MyBlock.Grid.Cell[cell_index_i-1][cell_index_j][cell_index_k].V;
  Jacobian[stencil_TOP]    = Jacobian[stencil_TOP]/MyBlock.Grid.Cell[cell_index_i][cell_index_j][cell_index_k+1].V;
  Jacobian[stencil_BOTTOM] = Jacobian[stencil_BOTTOM]/MyBlock.Grid.Cell[cell_index_i][cell_index_j][cell_index_k-1].V;
}

template<typename SOLN_pSTATE, typename SOLN_cSTATE>
void Initialize_Parameters(Hexa_Multi_Block<Hexa_Block<SOLN_pSTATE, SOLN_cSTATE> > &Local_Solution_Blocks, 
			   AdaptiveBlock3D_List 			&Local_Adaptive_Block_List, 
			    AdaptiveBlock3D_ResourceList                &Global_List){ 
/*****************************************************************************
 *  Assign values to Global Variables.                                        *
*****************************************************************************/   
  int counter = 0;
  int val;
	 	  
	  //std::cout<<" Yes, Jacobian Stencil Size is: "<<Jacobian_stencil_size<<std::endl;
	  Jacobian_Data = new DenseMatrix[Jacobian_stencil_size];
	  TempMat = new DenseMatrix[Jacobian_stencil_size];
	  numProcs = 1;
	  global_numBlk = Global_List.Nused;
	  
	  	 //====================================================================================
// 	  A hack to get number of processors that most likely won't work in AMR. 
// 	   int NUMPROC = global_numBlk % availProcs;
// 	   if (availProcs > global_numBlk){
// 	     numProcs = NUMPROC;
// 	   }
// 	   else //if (availProcs == global_numBlk)
// 	     numProcs =  availProcs;
	 //====================================================================================  
	   // this works, and works very well
	   LocalBlocksinProc = 0;
	   for (int i = 0; i< Global_List.Ncpu; i++){
	    if (MyPID == i){
	       if (Local_Solution_Blocks.Block_Used[0])
		// if (Local_Adaptive_Block_List.Nblk !=0 )
		numProcsUsed = 1;
		for (int j =0; j < Local_Adaptive_Block_List.Nblk; j++){
		  if (Local_Solution_Blocks.Block_Used[j]){
		     LocalBlocksinProc = LocalBlocksinProc + 1;
		  }
		 }
	     }
	   }
	  //==================================================================================== 
	   // execute a send and receive to establish number of Processors and number of PrevBlocks
	   
	   // from CFFC/MPI/MPI.h:------------- Returns summation of all integers sent by each processor.
// 	      inline int CFFC_Summation_MPI(const int send_value) {
// 		int receive_value;
// 		receive_value = send_value;
// 	      #ifdef _MPI_VERSION
// 		MPI::COMM_WORLD.Allreduce(&send_value, &receive_value, 1, MPI::INT, MPI::SUM);
// 	      #endif
// 		return (receive_value);
// 	      }
	   int mpi_sum_procs;
	   
	    for (int i = 0; i< Global_List.Ncpu; i++){
		if (MyPID == i){
		     numProcs = CFFC_Summation_MPI(numProcsUsed);
		     //MPI::COMM_WORLD.Allreduce(&send_value, &receive_value, 1, MPI::INT, MPI::SUM);
		}
	    }
// 	   if (MyPID ==0){
// 	      std::cout<<"Number of Procs used by the blocks is : "<<numProcs<<endl;
// 	   }
	  //====================================================================================
	   BlockList.resize(numProcs);
	  // CFFC has already determined the distribution of blocks over all the available procs
	   if (Local_Solution_Blocks.Block_Used[0]){
	      counter = 0;
	      NCi = Local_Solution_Blocks.Soln_Blks[counter].NCi;
	      NCj = Local_Solution_Blocks.Soln_Blks[counter].NCj;
	      NCk = Local_Solution_Blocks.Soln_Blks[counter].NCk;
	      Nghost = Local_Solution_Blocks.Soln_Blks[counter].Nghost;
	      blocksize = Local_Solution_Blocks.Soln_Blks[counter].NumVar();
	      Ni = NCi-2*Nghost;
	      Nj = NCj-2*Nghost;
	      Nk = NCk-2*Nghost;
	      ICl=Local_Solution_Blocks.Soln_Blks[counter].ICl;
	      JCl=Local_Solution_Blocks.Soln_Blks[counter].JCl;
	      KCl=Local_Solution_Blocks.Soln_Blks[counter].KCl;
	      ICu=Local_Solution_Blocks.Soln_Blks[counter].ICu;
	      JCu=Local_Solution_Blocks.Soln_Blks[counter].JCu;
	      KCu=Local_Solution_Blocks.Soln_Blks[counter].KCu;
	  }
 	  
// }
	  N = Ni;    //  i.e. the N is actually 5 * Number of cells,  each "entry" is actually 5 by 5
	  M = Nj;
	  O = Nk;

	  if (MyPID== 0){
		std::cout<<"N is "<<N<<endl; 
		std::cout<<"M is "<<M<<endl; 
		std::cout<<"O is "<<O<<endl; 
		std::cout<<"(ICl,ICu) is ("<<ICl<<","<<ICu<<")"<<endl;
		std::cout<<"(JCl,JCu) is ("<<JCl<<","<<JCu<<")"<<endl;
		std::cout<<"(KCl,KCu) is ("<<KCl<<","<<KCu<<")"<<endl;
		std::cout<<"Nghost is "<<Nghost<<endl; 
// 		std::cout<<"Blockize as known Globally is "<<blocksize<<endl;
	  } 
	  
	  for (int i = 0; i<numProcs; i++){    // BlockList stores the number of Blocks in each Proc
	    if (MyPID == i){
		BlockinProc = LocalBlocksinProc;
		BlockList[i] = LocalBlocksinProc;
// 		std::cout<<"Blocks in the Proc "<<i<<" are: "<<BlockinProc<<endl;
	    }
	  }
		 
	  MPI_Status state;
	  int tag = 7;
	  int addBlocks = 0;
	  std::vector <int> recvBlocks;
	  
	  for (int i = 0; i<numProcs; i++){
	    
	    if (i == 0){
		addBlocks = 0;
	    }
	    if (i > 0){
		if (MyPID == i){
		    recvBlocks.resize(i);
		    for (int j=0; j < i; j++){
		      MPI_Recv(&recvBlocks[j], 1, MPI_INT ,j,tag, MPI::COMM_WORLD ,&state);  // j = source
		      //MPI_Recv(void* data, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm communicator,  MPI_Status* status)

		    }
		}    
		if (MyPID < i){
		    MPI_Send(&LocalBlocksinProc, 1, MPI_INT ,i,tag, MPI::COMM_WORLD);   // i = destination
		    //MPI_Send( void* data, int count, MPI_Datatype datatype, int destination, int tag,  MPI_Comm communicator)
		}
	    }
	  }
	  
	  for (int i = 0; i<numProcs; i++){
	    if (MyPID == i){
		for (int k = 0; k < i; k++){
		    addBlocks = addBlocks + recvBlocks[k];
		}
		prevBlock = addBlocks;
	    }
	  }  

	Cummul_Blocks.resize(numProcs);
	int TempCummBlocks[numProcs];
	int TempBlockList[1];
	TempBlockList[0] = BlockList[MyPID];
	
	MPI_Allgather(TempBlockList,1,MPI_INT,TempCummBlocks ,1,MPI_INT, MPI::COMM_WORLD);
// 	std::cout<<"Gather operation is complete. Displaying results: "<< endl;

	int counted = 0;
	for (int i = 0; i<numProcs; i++){
	    if (MyPID == i){
	      for (int j = 0; j<numProcs; j++){
		  counted = counted + TempCummBlocks[j];
		  Cummul_Blocks[j] = counted;
	      }
	    }
	}

// 	  for (int i = 0; i<numProcs; i++){
// 	      std::cout<<""<<endl;
// 	      if (MyPID==i){
// 		std::cout<<"PID "<<MyPID<<" now has the following info: "; 
// 		for (int j = 0; j<numProcs; j++){
// 		  std::cout<<Cummul_Blocks[j]<<" ";
// 		}
// 	      }
// 	      std::cout<<""<<endl;
// 	  }
  
	  NumGlobalElements = global_numBlk * blocksize*(N)*(M)*(O);           // global dimension of the problem
	                       			// local dimension of the problem
	  MyGlobalElements = new global_ordinal_type[numProcs];
	  ProcList.resize(numProcs,0);
	  indx.resize(numProcs+1,0);
	  indx[0]=0;
	  int cumm = 0;
	  
	  for(int i=0; i<numProcs; i++){
	    ProcList[i] = BlockList[i] * blocksize*(N)*(M)*(O);    // prevBlock may be better suited, to allocate the Proc slot after the previous n-1 Procs!
	    if (i == (numProcs - 1) ){ 
	      ProcList[i] = NumGlobalElements - cumm;
	    }
	    cumm = cumm + ProcList[i];
	    indx[i+1] = prevBlock * blocksize*(N)*(M)*(O); //indx[i] + ProcList[i];
 
	    for (int j=0; j<numProcs; j++){
	      if (MyPID==j){
		MyElements = BlockList[j] * blocksize*(N)*(M)*(O) ; // to show the blockCount per Proc		 //ProcList[j];  		
		MyGlobalElements = new int[MyElements];
		for(int a=indx[j]; a< indx[j]+MyElements; a++){
		  MyGlobalElements[a-indx[j]]=a;
		}
	      }
	    }
	  }
	  
	 Map2 = new Epetra_Map(-1,MyElements,MyGlobalElements,0,*Comm);
	 AA = new Epetra_CrsMatrix(Copy, *Map2, (7)*blocksize);	 //Max number of entries per row is (7 + 7) * blocksize
// 	 std::cout<<"This is Proc number "<<MyPID<<" and here is my contribution to Matrix A: "<<*AA<<endl;
}

template<typename SOLN_pSTATE, typename SOLN_cSTATE>
void Initialize_Matrix(	Hexa_Multi_Block<Hexa_Block<SOLN_pSTATE, SOLN_cSTATE> > 	&Local_Solution_Blocks, 
			AdaptiveBlock3D_List 						&Local_Adaptive_Block_List,
			AdaptiveBlock3D_ResourceList                			&Global_Adaptive_Block_List, 
			Input_Parameters<SOLN_pSTATE, SOLN_cSTATE> 			&Input){ 

  int *block_i = new int[Jacobian_stencil_size];
  int *block_j = new int[Jacobian_stencil_size]; 
  int varcount = 0;
  
  int ii, jj, kk;                               // indices to be used by the Epetra Matrix
  double sumMat=0; 
  double absSumMat = 0;
  int countt = 0;
  int BlkN, BlkS, BlkE, BlkW, BlkT, BlkB, nBlkN, nBlkS, nBlkE, nBlkW, nBlkT, nBlkB;
  int cpuN, cpuS, cpuE, cpuW, cpuT, cpuB;
  int iNeigh, jNeigh, kNeigh;
  int ival, jval, kval;
  int CellID, neighCellID;
  int neighrow, neighcol;
  int BlkUsed, loop, cpudel, cpuUsed, prBlockUsed, xtra;
  int localBlknum;
  
  
  
    for (int i = 0; i < numProcs; i++){
//     int i = 1;
      if (MyPID == i){ 
    
	    const int storeVal = prevBlock;
   
		/****************************************************************************************************************************************************
		* Loop over blocks, load the Jacobian_Data from the Flux Jacobian Matrix (dR/dU) and Call Epetra Fill Matrix                                    *
		********************************************************************************************************************************/
// 		  std::cout<<"PID number "<<MyPID<<" is in Initialize Matrix and made it to stage 1"<< endl;
		  for(int l=0; l<Jacobian_stencil_size; l++) {
		    Jacobian_Data[l] = DenseMatrix(blocksize,blocksize,ZERO);   // Setting values of Jacobian_Data to null
		    TempMat[l] = DenseMatrix(blocksize,blocksize,ZERO);   // Setting values of TempMat to null
		  }
		  for(int m=0; m<Jacobian_stencil_size; m++) {
		    Initialize_TempM<SOLN_pSTATE, SOLN_cSTATE>(TempMat[m]);
		  }
 
		for (int  loop = 0; loop < LocalBlocksinProc; loop++){ // looping over the local number of blocks
		  
		   localBlknum = Local_Adaptive_Block_List.Block[loop].info.blknum;  // tell me the index of the local block
		   
		   prevBlock = storeVal + localBlknum;   // rightly so, and gets updated within the loops
  
		  // check who the neighbors are, and then check which neighbor block cells are in contact
		    BlkT = Local_Adaptive_Block_List.Block[loop].infoNeighbour(0,0,1)->blknum;
		    BlkB = Local_Adaptive_Block_List.Block[loop].infoNeighbour(0,0,-1)->blknum;
		    BlkN = Local_Adaptive_Block_List.Block[loop].infoNeighbour(0,1,0)->blknum;
		    BlkS = Local_Adaptive_Block_List.Block[loop].infoNeighbour(0,-1,0)->blknum;
		    BlkE = Local_Adaptive_Block_List.Block[loop].infoNeighbour(1,0,0)->blknum;
		    BlkW = Local_Adaptive_Block_List.Block[loop].infoNeighbour(-1,0,0)->blknum;
		    // find the Proc ID for the neighbor block
		    cpuT = Local_Adaptive_Block_List.Block[loop].infoNeighbour(0,0,1)->cpu;
		    cpuB = Local_Adaptive_Block_List.Block[loop].infoNeighbour(0,0,-1)->cpu;
		    cpuN = Local_Adaptive_Block_List.Block[loop].infoNeighbour(0,1,0)->cpu;
		    cpuS = Local_Adaptive_Block_List.Block[loop].infoNeighbour(0,-1,0)->cpu;
		    cpuE = Local_Adaptive_Block_List.Block[loop].infoNeighbour(1,0,0)->cpu;
		    cpuW = Local_Adaptive_Block_List.Block[loop].infoNeighbour(-1,0,0)->cpu;
		    // How many cells are adjacent to cell center 0?
		    nBlkT = Local_Adaptive_Block_List.Block[loop].nT;
		    nBlkB = Local_Adaptive_Block_List.Block[loop].nB;
		    nBlkN = Local_Adaptive_Block_List.Block[loop].nN;
		    nBlkS = Local_Adaptive_Block_List.Block[loop].nS;
		    nBlkE = Local_Adaptive_Block_List.Block[loop].nE;
		    nBlkW = Local_Adaptive_Block_List.Block[loop].nW;


		    for(int i= ICl; i<=ICu; i++){    
		      for(int j= JCl; j<=JCu; j++){  
			for(int k= KCl; k<=KCu; k++){ 
			  // indices to be used by the Epetra Matrix
			  ii=i-ICl;
			  jj=j-JCl;
			  kk=k-KCl;
			  
			  NewGet_Block_Index<SOLN_pSTATE, SOLN_cSTATE>(ii,jj,kk, block_i, block_j, 
								       BlkN, BlkS, BlkE, BlkW, BlkT, BlkB, 
								       nBlkN, nBlkS, nBlkE, nBlkW, nBlkT, nBlkB, 
								       cpuN, cpuS, cpuE, cpuW, cpuT, cpuB, ii, jj, kk);
			  
			  CellID = ii + jj*N + kk * N * M;
			  
			  switch(Input.i_Flux_Function){
			    
			    case FLUX_FUNCTION_HLLE:
			      First_Order_Inviscid_Jacobian_HLLE<SOLN_pSTATE,SOLN_cSTATE>(i,j,k,Jacobian_Data,Local_Solution_Blocks.Soln_Blks[loop]);  
			      //case SECOND_ORDER_VISCOUS_WITH_HLLE:
			      //Second_Order_Viscous_Jacobian(i,j,k, Jacobian_Data);   
			      break;
			    
			    case FLUX_FUNCTION_ROE:
			      First_Order_Inviscid_Jacobian_Roe<SOLN_pSTATE,SOLN_cSTATE>(i,j,k,Jacobian_Data,Local_Solution_Blocks.Soln_Blks[loop]);
			       /// case SECOND_ORDER_VISCOUS_WITH_ROE :	
			       //  First_Order_Inviscid_Jacobian_Roe<SOLN_pSTATE,SOLN_cSTATE>(i,j,k,Jacobian_Data,Local_Solution_Blocks.Soln_Blks[loop]);
			       //  Second_Order_Viscous_Jacobian(i,j,k, Jacobian_Data);   
			      break;
			  default:
			    cerr<<"Please choose either HLLE or Roe"; exit(1);
			    break;
			  }			      
			      
	   /*            //   The other menu options would be:		   
			  if (strcmp(Flux_Function_Type, "Godunov") == 0) {i_Flux_Function = FLUX_FUNCTION_GODUNOV;} 
			  else if (strcmp(Flux_Function_Type, "Roe") == 0) {i_Flux_Function = FLUX_FUNCTION_ROE;}
			  else if (strcmp(Flux_Function_Type, "Rusanov") == 0) {i_Flux_Function = FLUX_FUNCTION_RUSANOV;} 
			  else if (strcmp(Flux_Function_Type, "HLLE") == 0) {i_Flux_Function = FLUX_FUNCTION_HLLE;} 
			  else if (strcmp(Flux_Function_Type, "Linde") == 0) {i_Flux_Function = FLUX_FUNCTION_LINDE;} 
			  else if (strcmp(Flux_Function_Type, "HLLC") == 0) {i_Flux_Function = FLUX_FUNCTION_HLLC;} 
			  else if (strcmp(Flux_Function_Type, "AUSM_plus_up") == 0) {i_Flux_Function = FLUX_FUNCTION_AUSM_PLUS_UP;} 
			  else if (strcmp(Flux_Function_Type, "Lax_Friedrichs") == 0) {i_Flux_Function = FLUX_FUNCTION_LAX_FRIEDRICHS;} 
			  else {i_command = INVALID_INPUT_VALUE;} // endif  
	  */		      

			  for( int block = 0; block < Jacobian_stencil_size; block++){
			    if (block_j[block] != -1){                  // block_i is the row, block_j is the column
			      
// 			     std::cout<<"(row,col) to send to Fillmatrix is ("<<block_i[block]<<","<<block_j[block]<<")"<<endl;

			      FillMatrix<SOLN_pSTATE, SOLN_cSTATE> (block_i[block], block_j[block], Jacobian_Data[block]);  // modify on j
			    }
			      Jacobian_Data[block].zero(); //Just in case to avoid +=/-= issues
			  } 
		
			}
		      }
		    }
		
		}   // looping over local blocks++++++++++++++ END OF BLOCK LOOP +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
      }  // "if MyPID == i loop "****************** END OF MyPID == 1 conditional *****************************************************
      
    } // loop for the Procs========= END OF PROC LOOP ==================================================

    AA->FillComplete();
    
    // Write to file Matrix Sparsity and Epetra matrix
    
    delete[] Jacobian_Data; delete[] block_i; delete[] block_j;  delete[] TempMat; 

}

template<typename SOLN_pSTATE, typename SOLN_cSTATE>
void Initialize_Vector(Hexa_Multi_Block<Hexa_Block<SOLN_pSTATE, SOLN_cSTATE> > &Local_Solution_Blocks, 
		       AdaptiveBlock3D_List 			&Local_Adaptive_Block_List, 
		    AdaptiveBlock3D_ResourceList                &Global_Adaptive_Block_List){ 
/*******************************************************************************************************************
 *  Assign the RHS Sensitivity Vector via evaluating based on a functional                         *
 *******************************************************************************************************************/   
  
  bb = new Epetra_Vector(*Map2);
//   std::cout<<"This is Proc number "<<MyPID<<" and here is my contribution to vector b: "<<*bb<<endl;

  
  double val, gamma;
  int index, ErrVec;
  gamma = 1.4;
  int h;
  
  for (int i = 0; i<numProcs; i++){
	if (MyPID == i){
	      for (int a = 0; a<MyElements; a++){
		h = MyGlobalElements[a];
		if ( (h+1) % blocksize == 0){
		  val = (gamma-1);
		}
		else{
		  val = 0.0;
		}
		index = h;
// 	      	std::cout<<"PID is : "<<MyPID<<"  MyElements is: "<<MyElements<<" index position is "<<h<<"and blocksize is: "<<blocksize<<endl;
		ErrVec = bb->SumIntoGlobalValues(1,0,&val,&index);
		if (ErrVec !=0){
		      std::cout<<"Epetra Vector: I have an error located on line "<<h<<" of value= "<<ErrVec<<std::endl;
		      exit(123);
		}
	      }
	}
  }
 

}

template<typename SOLN_pSTATE, typename SOLN_cSTATE>
void Solve_System(Hexa_Multi_Block<Hexa_Block<SOLN_pSTATE, SOLN_cSTATE> > &Local_Solution_Blocks, 
		  AdaptiveBlock3D_List 				&Local_Adaptive_Block_List, 
		  AdaptiveBlock3D_ResourceList                	&Global_Adaptive_Block_List,
		  Input_Parameters<SOLN_pSTATE, SOLN_cSTATE> 	&Input){ 
/********************************************************************************************************************************************************
 *  Aztec Solve                                *
 *******************************************************************************************************************************************************/     
//   x = new Epetra_Vector(*Map2);
 
  Epetra_Vector x(*Map2);
  int CellID;
  int tot = N*M*O;
  
// you can get the Transpose here ====================================================== 
//   if (Transpos == 1){
// 	// AA -> SetUseTranspose(true);  // wrong approach, should flip at the writing of J
//   } else{
// 	// AA -> SetUseTranspose(false);
//   }
  
//======================================================================================  
  
// Solve Aztec system--------------------------------------------------------------------------
  Epetra_LinearProblem problem(&*AA, &x, &*bb);
    AztecOO solver(problem); 
    int Niters=100;
    solver.SetAztecOption(AZ_kspace, Niters);
    // Krylov Solver
    solver.SetAztecOption(AZ_solver, AZ_gmres_condnum);
  //   solver.SetAztecOption(AZ_precond, AZ_Jacobi);
    solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
    solver.SetAztecOption(AZ_subdomain_solve,AZ_ilu);
    solver.SetAztecOption(AZ_diagnostics,AZ_all);
    solver.SetAztecParam(AZ_tol, 1.e-10);
    solver.SetAztecParam(AZ_graph_fill,3);
    solver.Iterate(2000, 1.0E-10);
//   // LU direct solve
//   solver.SetAztecOption(AZ_solver, AZ_lu);
//   solver.SetAztecOption(AZ_subdomain_solve,AZ_lu);
//   solver.SetAztecParam(AZ_graph_fill,1);
  
  int TotalPrevBlocks = 0;
  int X_vect = 0;
  int BlkN, BlkS, BlkE, BlkW, BlkT, BlkB, nBlkN, nBlkS, nBlkE, nBlkW, nBlkT, nBlkB;
  int cpuN, cpuS, cpuE, cpuW, cpuT, cpuB;
  int ii,jj,kk;
  
//   std::cout<<x<<endl;
  
  std::stringstream stream1;
     
      if (Transpos == 1){
	    stream1 << "Adjoint_soln_Proc_"<<MyPID<<"_Transpose.txt";
      } else{
	    stream1 << "Adjoint_soln_Proc_"<<MyPID<<"_NoTransp.txt";
      }
      std::string fileName1 = stream1.str();
      FILE* fout = NULL;
      
      if (WriteMatrixtoFile == 1){     
	FILE* fout = fopen(fileName1.c_str(), "w");
      }
	// Writing the Epetra vector to file
	  for (int loop = 0; loop < LocalBlocksinProc; loop++){
	      for (int i =ICl; i <= ICu; i++){
		  for (int j =JCl; j <= JCu; j++){
		    for (int k =KCl; k <= KCu; k++){
		      
		      ii = i-ICl;
		      jj = j-JCl;
		      kk = k-KCl;
		    
		      CellID = ii + jj*N + kk*N*M;
// 		      CellID = i + j*N + k*N*M;
		      X_vect = (loop* N*M*O + CellID) * blocksize; //TotalPrevBlocks * N*M*O * blocksize + CellID; 

		      Local_Solution_Blocks.Soln_Blks[loop].AdjointSoln[i][j][k].rho = x[X_vect + 0];
		      Local_Solution_Blocks.Soln_Blks[loop].AdjointSoln[i][j][k].v.x = x[X_vect + 1];
		      Local_Solution_Blocks.Soln_Blks[loop].AdjointSoln[i][j][k].v.y = x[X_vect + 2];
		      Local_Solution_Blocks.Soln_Blks[loop].AdjointSoln[i][j][k].v.z = x[X_vect + 3];
		      Local_Solution_Blocks.Soln_Blks[loop].AdjointSoln[i][j][k].p   = x[X_vect + 4];
		      
		      // center coordinates for that cell
		      double x = Local_Solution_Blocks.Soln_Blks[loop].Grid.Cell[i][j][k].Xc.x;
		      double y = Local_Solution_Blocks.Soln_Blks[loop].Grid.Cell[i][j][k].Xc.y;
		      double z = Local_Solution_Blocks.Soln_Blks[loop].Grid.Cell[i][j][k].Xc.z;
		      
		      // output density value
		      double rho1 = Local_Solution_Blocks.Soln_Blks[loop].AdjointSoln[i][j][k].rho;
		      if (WriteMatrixtoFile == 1){  
			fprintf(fout, "%e\t %e\t %e\t %e\t ", x, y, z, rho1); 
			fprintf(fout, "\n");
		      }
		    }
		  }
	      }
	      Local_Solution_Blocks.Output_Adjoint_Tecplot(Input, Local_Adaptive_Block_List, 0, 1.1);	
	  }
     if (WriteMatrixtoFile == 1){  	  
	fclose(fout);
	std::cout<<fileName1.c_str()<<" successfully written by PID "<<MyPID<<endl;
     }
    
 // From Hexa MultiBlock.h : int Output_Adjoint_Tecplot(Input_Parameters<typename HEXA_BLOCK::Soln_pState, typename HEXA_BLOCK::Soln_cState> &Input, AdaptiveBlock3D_List &Local_Adaptive_Block_List,
 
 
 //===========================================================================
 
    if (WriteVector == 1){
      std::stringstream stream2;
      for (int a=0; a<numProcs; a++){
	if (MyPID == a){
    //       std::cout<<"Proc "<<a<<" has "<<tot*blocksize<<" Elements"<<endl; 
	  if (Transpos == 1){
	    stream2 << "x_Proc_"<<MyPID<<"_Transpose.txt";
	  } else{
	    stream2 << "x_Proc_"<<MyPID<<"_NoTransp.txt";
	  }
	  std::string fileName2 = stream2.str();
	  FILE* fout = fopen(fileName2.c_str(), "w");
	  for (int j = 0; j < MyElements/*tot*blocksize*/; j++){ 
	    fprintf(fout, "%e", x[j]); 
	    fprintf(fout, "\n");
	  } 
	  fclose(fout);
	}
      }
    }
    
    if (WriteMatrixtoFile == 1){
	  std::stringstream stream14;
	  std::string fileName3;
	  boost::posix_time::ptime ComputerTime = boost::posix_time::microsec_clock::local_time();
	  if (Transpos == 1){
// 	    stream14 << "A_global_size_"<<N<<"_cubed_Proc"<<MyPID<<"_Time_"<<ComputerTime.time_of_day()<<".txt";
	    stream14 << "A_global_size_"<<N<<"_cubed_Proc"<<MyPID<<"_Transpose.txt";

	  }
	  else{
	    stream14 << "A_global_size_"<<N<<"_cubed_Proc"<<MyPID<<"_NoTransp.txt";
	  }
	  fileName3 = stream14.str();
	  std::ofstream Outfile2;
	  Outfile2.open(fileName3.c_str(), std::ios::out );
	  if (Outfile2.is_open()) {
	    Outfile2 << *AA << endl;
	  }
	  Outfile2.close();
	  std::cout<<fileName3.c_str()<<" successfully written by PID "<<MyPID<<endl;
    }
    
    if (PrintSparse == 1){
      if (Transpos == 1){
	  Ifpack_PrintSparsity(*AA, "mat_Transp.ps");
      } else{
	  Ifpack_PrintSparsity(*AA, "mat_NOTransp.ps");
      }
       
    }
 
}



template<typename SOLN_pSTATE, typename SOLN_cSTATE>
int Adjoint_Problem(Hexa_Multi_Block<Hexa_Block<SOLN_pSTATE, SOLN_cSTATE> > &Local_Solution_Blocks, 
		    AdaptiveBlock3D_List 			&Local_Adaptive_Block_List,
// 		    AdaptiveBlock3D_Info 			&InfoBlock, 
		    AdaptiveBlock3D_ResourceList                &Global_Adaptive_Block_List,
		    Input_Parameters<SOLN_pSTATE, SOLN_cSTATE> 	&Input){ 
/**********************************************************************************************************************************************************
 *  Main function that gets called from AMR.h, and calls the other Initialize functions, which have other subfunctions                                       *
 *********************************************************************************************************************************************************/     
  Comm = new Epetra_MpiComm( MPI::COMM_WORLD );
  availProcs = Comm->NumProc();
  MyPID = Comm->MyPID();
	  
  if (MyPID ==0){
	std::cout<<"..."<<endl<<endl<<endl;
	std::cout<<"Calling adjoint solver "<<std::endl;
    }	
	  
	  // Ni, Nj, Nk and blocksize belong below here
	  Initialize_Parameters(Local_Solution_Blocks, Local_Adaptive_Block_List, Global_Adaptive_Block_List);

	  // Assemble and fill in Epetra Matrix
	  Initialize_Matrix(Local_Solution_Blocks, Local_Adaptive_Block_List, /*&InfoBlock,*/ Global_Adaptive_Block_List, Input);

// 	  // Assemble and fill in Epetra Vector
	  Initialize_Vector(Local_Solution_Blocks, Local_Adaptive_Block_List, Global_Adaptive_Block_List);
// 
// 	  // Call Aztec or Belos solver
	  Solve_System(Local_Solution_Blocks, Local_Adaptive_Block_List, Global_Adaptive_Block_List, Input);
	  
	  
	  
      delete [] MyGlobalElements;
	
	std::cout<<"leaving Adjoint_Problem"<<std::endl;
	  delete bb; bb=NULL; delete AA; AA=NULL; delete Map2; Map2 = NULL; delete Comm; Comm=NULL;
	
// 	MPI_Finalize() ;
	return 0;

}

 