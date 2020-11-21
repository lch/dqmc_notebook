module SVDM
using LinearAlgebra
import ..Hubbard_Model: Mat_Type
import Base.copyto!

mutable struct SVD_Store{T <: Real}
	U::Matrix{T}
	D::Vector{T}
	V::Matrix{T}
end

function SVD_Store(T::Type, MatDim::Int)
	SVD_Store{T}(zeros(T, MatDim, MatDim), 
		zeros(T, MatDim), 
		zeros(T, MatDim, MatDim)
	)
end

# This is a naive QR method, and it might behave bad in large U. 
# But it should satisfy most case when considering efficiency.
function qr_svd(A::Matrix{Mat_Type})
	U, R = LinearAlgebra.qr(A)
	D = LinearAlgebra.Diagonal(R)
	V = inv(D) * R
	Matrix{Mat_Type}(U), D.diag, V
end

function svd_wrap(A::Matrix{Mat_Type})::SVD_Store
	SVD_Store(qr_svd(A)...)
end

function qr_udv!(A::Matrix{Mat_Type}, ret::SVD_Store)
	ret.U, ret.V = qr(A)
	ret.D = diag(ret.V)
	lmul!((Diagonal(1 ./ ret.D)), ret.V)
end

function svd_wrap!(A::Matrix{Mat_Type}, ret::SVD_Store)
	qr_udv!(A, ret)
end

function init_b_udv_store!(udv_store::Vector{SVD_Store{Mat_Type}}, N_ns::Int, MatDim::Int, ::Val{:β_τ})
	eye = Matrix{Mat_Type}(Diagonal(ones(MatDim)))
	for i = 1:N_ns+1
		udv_store[i] = SVD_Store(Mat_Type, MatDim)
	end
	copyto!(udv_store[N_ns+1].U, eye)
	udv_store[N_ns+1].D = ones(MatDim)
	copyto!(udv_store[N_ns+1].V, eye)
end

function init_b_udv_store!(udv_store::Vector{SVD_Store{Mat_Type}}, N_ns::Int, MatDim::Int, ::Val{:τ_0})
	eye = Matrix{Mat_Type}(Diagonal(ones(MatDim)))
	for i = 1:N_ns+1
		udv_store[i] = SVD_Store(Mat_Type, MatDim)
	end
	copyto!(udv_store[1].U, eye)
	udv_store[1].D = ones(MatDim)
	copyto!(udv_store[1].V, eye)
end

@inline function udv_index(time_index::Int, N_ns_int::Int)
	cui = time_index ÷ N_ns_int + 1
	if time_index % N_ns_int == 0
		cui -= 1
	end
	return cui
end

function copyto!(A::SVD_Store{T}, B::SVD_Store{T}) where T <: Number
	copyto!(A.U, B.U)
	copyto!(A.D, B.D)
	copyto!(A.V, B.V)
end

end