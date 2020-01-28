#import subprocess
import sys
import numpy as np
#import scipy.optimize as scipop
#import scipy as sc
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures
from bandwidth_functions import t_copy_vec_1d, t_copy_vec_2d
#from transpose_functions import t_dtranspose
#from dgemm_functions import t_dgemm_cpu, t_dgemm_gpu

from dataclasses import dataclass
from enum import Enum

class Buffer_type(Enum):
    R = 0
    W = 1
    RW = 2

class Memory_location(Enum):
    Unallocated = -2 
    Host = -1
    Device = 0

class Memory_layout(Enum):
    Row_major = 0
    Col_major = 1


@dataclass
class Buffer2d:
    row_sz: int
    col_sz: int
    update_grid: np.ndarray
    mode: Buffer_type
    layout: Memory_layout
    elems_updated = 0
    loc: Memory_location = Memory_location.Unallocated
    elem_sz : int = 8
    #father: Buffer2d
    def __init__(self,row_dim, col_dim, md, layout): 
        self.row_sz = row_dim
        self.col_sz = col_dim
        if layout == Memory_layout.Row_major:  
            self.update_grid = np.full((row_dim,col_dim), -1)
        else:  
            self.update_grid = np.full((col_dim,row_dim), -1)
        self.mode = md
        self.layout = layout

    def allocate(self, loc): 
        self.loc = loc
        if self.layout == Memory_layout.Row_major:  
            self.update_grid = np.full((self.row_sz,self.col_sz), 0)
        else:  
            self.update_grid = np.full((self.col_sz,self.row_sz), 0)
        return 0 # t_allocate(self.row_sz*self.col_sz*self.elem_sz, loc)Here there should be time for buffer allocation?

    def initialize(self): 
        if self.mode == Buffer_type.W:
            sys.exit('initialize: error -> Initalizing Write buffer')       
        if self.layout == Memory_layout.Row_major:  
            self.update_grid = np.full((self.row_sz,self.col_sz), 1)
        else:  
            self.update_grid = np.full((self.col_sz,self.row_sz), 1)
        return 0

    def transfer_to(self, Buffer2d_target, row_dim, col_dim, offset_row, offset_col): 
        if Buffer2d_target.mode == Buffer_type.W:
            sys.exit('transfer_to: error -> Trying to transfer to Write buffer')  
        if self.loc == Memory_location.Unallocated:
            sys.exit('transfer_to: error -> src buffer is unallocated') 
        if Buffer2d_target.loc == Memory_location.Unallocated:
            sys.exit('transfer_to: error -> dest buffer is unallocated') 
        if self.update_grid[0][0] == 0:
            print('transfer_to: warning -> src buffer is uninitialized')
        if Buffer2d_target.elem_sz!= self.elem_sz:
            sys.exit('transfer_to: error -> Buffers are of different type')
        if Buffer2d_target.layout!= self.layout:
            sys.exit('transfer_to: error -> Not implemented yet, change it')
        elif self.layout == Memory_layout.Row_major:
            dim1, dim2, ldim = row_dim, col_dim, self.col_sz
            offdim1, offdim2 = offset_row, offset_col
            target_dim1, target_dim2 = Buffer2d_target.row_sz, Buffer2d_target.col_sz
        else:
            dim1, dim2, ldim = col_dim, row_dim, self.row_sz
            offdim1, offdim2 = offset_col, offset_row
            target_dim1, target_dim2 = Buffer2d_target.col_sz, Buffer2d_target.row_sz 
        if target_dim1 < offdim1 + dim1 or target_dim2 < offdim2 + dim2:
            sys.exit('transfer_to: error -> invalid transfer dims Offset(%d,%d) + transfer(%d,%d) > target(%d,%d)' % (offdim1, offdim2, dim1, dim2, target_dim1, target_dim2 ))
        if Buffer2d_target.elems_updated < target_dim1*target_dim2 or Buffer2d_target.mode == Buffer_type.RW:
            overlap_ctr = 0
            for dim1ctr in range(offdim1, dim1 + offdim1):
                for dim2ctr in range(offdim2, dim2 + offdim2):
                    temp = Buffer2d_target.update_grid[dim1ctr][dim2ctr]
                    if  temp == 0:
                        Buffer2d_target.update_grid[dim1ctr][dim2ctr] = 1
                        Buffer2d_target.elems_updated += 1
                    else:
                        overlap_ctr +=1
                        Buffer2d_target.update_grid[dim1ctr][dim2ctr] += 1         
            if (overlap_ctr == dim1*dim2 and Buffer2d_target.mode == Buffer_type.R):
                print('transfer_to: warning -> already updated, skipping update')
                return 0 
            elif overlap_ctr > 0 and Buffer2d_target.mode == Buffer_type.R:
                ratio = overlap_ctr*1.0/(dim1*dim2)
                print('transfer_to: warning -> partial overlap %.5lf, transfer time is not exact' %  ratio)
            else:
                ratio = 0
        else:
             print('transfer_to: warning -> Target fully updated, skipping update')
             return 0
        return (1-ratio)*t_copy_vec_2d(dim1, dim2, ldim, self.loc.value, Buffer2d_target.loc.value, self.elem_sz)

class gemm_buffers:
    A_host: Buffer2d
    A_device: Buffer2d
    B_host: Buffer2d
    B_device: Buffer2d
    C_host: Buffer2d
    C_device: Buffer2d
    trans_buffer: Buffer2d
    def __init__(self,M, N, K, layoutA, layoutB, layoutC): 
        self.A_host = Buffer2d(M,K, Buffer_type.R , Memory_layout(layoutA))
        self.B_host = Buffer2d(K,N, Buffer_type.R , Memory_layout(layoutB))
        self.C_host = Buffer2d(M,N, Buffer_type.RW , Memory_layout(layoutC))
        self.A_device = Buffer2d(M,K, Buffer_type.R , Memory_layout(layoutA))
        self.B_device = Buffer2d(K,N, Buffer_type.R , Memory_layout(layoutB))
        self.C_device = Buffer2d(M,N, Buffer_type.W , Memory_layout(layoutC))
        self.trans_buffer = Buffer2d(M,N, Buffer_type.RW , Memory_layout(layoutC))

    def allocate_host(self): 
        timer = 0
        timer += self.A_host.allocate(Memory_location.Host)
        timer += self.B_host.allocate(Memory_location.Host)
        timer += self.C_host.allocate(Memory_location.Host)
        timer += self.trans_buffer.allocate(Memory_location.Host)
        return timer

    def allocate_device(self):
        timer = 0
        timer += self.A_device.allocate(Memory_location.Device)
        timer += self.B_device.allocate(Memory_location.Device)
        timer += self.C_device.allocate(Memory_location.Device)
        return timer

    def initialize(self):
        self.A_host.initialize()
        self.B_host.initialize()

def t_bus_recv_kernel_data(M1, N1, M_offset, N_offset, gemm_buffers, printflag):
    t_get_C = gemm_buffers.C_device.transfer_to(gemm_buffers.C_host, int(M1), int(N1), M_offset,N_offset)
    if printflag:
        print('t_bus_recv_buffers(%d,%d) : t_total = %.5lf' % ( M1, N1, t_get_C))
    return t_get_C

def t_bus_send_kernel_data(M1, N1, K1, M_offset, N_offset, K_offset, gemm_buffers, printflag):
    t_transfer_A = gemm_buffers.A_host.transfer_to(gemm_buffers.A_device, int(M1), int(K1), M_offset,K_offset)
    t_transfer_B = gemm_buffers.B_host.transfer_to(gemm_buffers.B_device, int(K1), int(N1), K_offset,N_offset)
    t_transfer_C = 0 
    t_transfer_total  = t_transfer_A + t_transfer_B + t_transfer_C
    if printflag:
        print('t_bus_send_buffers(%d,%d,%d) : t_total = %.5lf (A=%.5lf s , B=%.5lf s,C=%.5lfs )' % ( M1, N1, K1, t_transfer_total, t_transfer_A, t_transfer_B,t_transfer_C))
    return t_transfer_total

#t_bus_send_kernel_data(1000, 1000, 1000, 0, 0 , 0, My_buffers, True)
#t_bus_recv_kernel_data(1000, 1000, 0 , 0, My_buffers, True)

