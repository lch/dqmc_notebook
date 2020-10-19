module SVDM
using LinearAlgebra
const Mat_Type = Float64

mutable struct SVD_Store
	U::Matrix{Mat_Type}
	D::Matrix{Mat_Type}
	V::Matrix{Mat_Type}
end

# This is a naive QR method, and it might behave bad in large U. 
# But it should satisfy most case when considering efficiency.
function qr_svd(A::Matrix{Mat_Type})
	U, R = LinearAlgebra.qr(A)
	D = LinearAlgebra.Diagonal(R)
	V = inv(D) * R
	Matrix(U), Matrix(D), V
end

function svd_wrap(A::Matrix{Mat_Type})::SVD_Store
	SVD_Store(qr_svd(A)...)
end

end