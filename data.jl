module Data

import ..SVDM: SVD_Store
import ..Hubbard_Model: Mat_Type


"""
## The data storage of B matrix multiplication

It is expensive when we calculate the Green's Function with all B matrix 
multiplication at each Numerical Stability time slice. It is obviously that 
when we sweep up, only B_τ_0 is required to update, and when sweep down, 
only B_β_τ is required to update. While one is update, the other is calculated
previously. 

Here is an example to explain how these multiplication is stored. 
Suppose we have 20 B matrices corresponding time slices, and Numerical 
Stability is performed every 5 time slices. Then, we'll have following scheme.

```
B_up_β_τ[5] = (I, I, I)				B_up_τ_0[5] = B_20 * ... * B_1
B_up_β_τ[4] = B_20 * ... * B_16		B_up_τ_0[4] = B_15 * ... * B_1
B_up_β_τ[3] = B_20 * ... * B_11		B_up_τ_0[3] = B_10 * ... * B_1
B_up_β_τ[2] = B_20 * ... * B_6		B_up_τ_0[2] = B_5 * ... * B_1
B_up_β_τ[1] = B_20 * ... * B_1		B_up_τ_0[1] = (I, I, I)
```
For example, the Green's Function G(5,5) = (1 + B_up_τ_0[2] * B_up_β_τ[2])^{-1}
Usually, there should be exactly N_ns elements to store. However, we here 
add one more elements and its value was set to I. The benefit is that we can
use same index number to calculate Green's Function.

"""
struct persistent{T <: Number}
	B_τ_0_up_udv::Vector{SVD_Store{T}}
	B_τ_0_dn_udv::Vector{SVD_Store{T}}
	B_β_τ_up_udv::Vector{SVD_Store{T}}
	B_β_τ_dn_udv::Vector{SVD_Store{T}}
	function persistent(T::Type, N_ns::Int, MatDim::Int)
		B_τ_0_up_udv = Vector{SVD_Store{T}}(undef, N_ns + 1)
		B_τ_0_dn_udv = Vector{SVD_Store{T}}(undef, N_ns + 1)
		B_β_τ_up_udv = Vector{SVD_Store{T}}(undef, N_ns + 1)
		B_β_τ_dn_udv = Vector{SVD_Store{T}}(undef, N_ns + 1)
		new{T}(B_τ_0_up_udv, B_τ_0_dn_udv, B_β_τ_up_udv, B_β_τ_dn_udv)
	end
end

struct temporary{T <: Number}
	mat::Matrix{T}
	exp_V::Vector{T}
	udv::SVD_Store{T}
	udv_up::SVD_Store{T}
	udv_dn::SVD_Store{T}
	b_inv::Matrix{T}
	function temporary(T::Type, MatDim::Int)
		mat_tmp = zeros(T, MatDim, MatDim)
		expV_tmp = zeros(T, MatDim)
		udv = SVD_Store(T, MatDim)
		udv_up = SVD_Store(T, MatDim)
		udv_dn = SVD_Store(T, MatDim)
		b_inv = zeros(T, MatDim, MatDim)
		new{T}(mat_tmp, expV_tmp, udv, udv_up, udv_dn, b_inv)
	end
end

end