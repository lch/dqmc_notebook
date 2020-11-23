module CoreM

using LinearAlgebra
using ..SVDM: SVD_Store, svd_wrap, svd_wrap!, udv_index
using ..Hubbard_Model: Mat_Type, expV, expV!, Param, Square_Lattice
using ..Data: temporary, persistent
using ..Obser: sampling

function B_up(auxF::Vector{Int}, exp_T::Matrix{Mat_Type})
	expV(1, auxF) * exp_T
end

function B_dn(auxF::Vector{Int}, exp_T::Matrix{Mat_Type})
	expV(-1, auxF) * exp_T
end

function B_up_inv(auxF::Vector{Int}, exp_mT::Matrix{Mat_Type})
	exp_mT * expV(-1, auxF)
end

function B_dn_inv(auxF::Vector{Int}, exp_mT::Matrix{Mat_Type})
	exp_mT * expV(1, auxF)
end

function B_up!(auxF::AbstractVector{Int}, exp_T::Matrix{Mat_Type}, exp_V_tmp::Vector{Mat_Type})
	expV!(1, auxF, exp_V_tmp)
	Diagonal(exp_V_tmp) * exp_T
end

function B_dn!(auxF::AbstractVector{Int}, exp_T::Matrix{Mat_Type}, exp_V_tmp::Vector{Mat_Type})
	expV!(-1, auxF, exp_V_tmp)
	Diagonal(exp_V_tmp) * exp_T
end

function B_up_inv!(auxF::AbstractVector{Int}, exp_mT::Matrix{Mat_Type}, exp_V_tmp::Vector{Mat_Type})
	expV!(-1, auxF, exp_V_tmp)
	exp_mT * Diagonal(exp_V_tmp)
end

function B_dn_inv!(auxF::AbstractVector{Int}, exp_mT::Matrix{Mat_Type}, exp_V_tmp::Vector{Mat_Type})
	expV!(1, auxF, exp_V_tmp)
	exp_mT * Diagonal(exp_V_tmp)
end

function B_up!(auxF::AbstractVector{Int}, exp_T::Matrix{Mat_Type}, exp_V_tmp::Vector{Mat_Type}, B_mat::AbstractArray{Mat_Type})
	expV!(1, auxF, exp_V_tmp)
	mul!(B_mat, Diagonal(exp_V_tmp), exp_T)
end

function B_dn!(auxF::AbstractVector{Int}, exp_T::Matrix{Mat_Type}, exp_V_tmp::Vector{Mat_Type}, B_mat::AbstractArray{Mat_Type})
	expV!(-1, auxF, exp_V_tmp)
	mul!(B_mat, Diagonal(exp_V_tmp), exp_T)
end

function B_up_inv!(auxF::AbstractVector{Int}, exp_mT::Matrix{Mat_Type}, exp_V_tmp::Vector{Mat_Type}, B_mat::AbstractArray{Mat_Type})
	expV!(-1, auxF, exp_V_tmp)
	mul!(B_mat, exp_mT, Diagonal(exp_V_tmp))
end

function B_dn_inv!(auxF::AbstractVector{Int}, exp_mT::Matrix{Mat_Type}, exp_V_tmp::Vector{Mat_Type}, B_mat::AbstractArray{Mat_Type})
	expV!(1, auxF, exp_V_tmp)
	mul!(B_mat, exp_mT, Diagonal(exp_V_tmp))
end

function init_B_mat_list(auxf::Matrix{Int}, exp_T::Matrix{Mat_Type}, MatDim::Int, N_time_slice::Int)
	B_up_list = Array{Mat_Type,3}(undef, MatDim, MatDim, N_time_slice)
	B_dn_list = Array{Mat_Type,3}(undef, MatDim, MatDim, N_time_slice)
	for i = 1:N_time_slice
		B_up_list[:,:,i] = B_up(auxf[:,i], exp_T)
		B_dn_list[:,:,i] = B_dn(auxf[:,i], exp_T)
	end
	return B_up_list, B_dn_list
end

function init_B_mat_list!(auxf::Matrix{Int}, exp_T::Matrix{Mat_Type}, tmp::temporary, MatDim::Int, N_time_slice::Int)
	B_up_list = Array{Mat_Type,3}(undef, MatDim, MatDim, N_time_slice)
	B_dn_list = Array{Mat_Type,3}(undef, MatDim, MatDim, N_time_slice)
	@views for i = 1:N_time_slice
		B_up_list[:,:,i] = B_up!(auxf[:,i], exp_T, tmp.exp_V)
		B_dn_list[:,:,i] = B_dn!(auxf[:,i], exp_T, tmp.exp_V)
	end
	return B_up_list, B_dn_list
end

function B_τ_0(time_index::Int, B_list::Array{Mat_Type,3}, MatDim::Int)::SVD_Store
	Btmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Utmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Dtmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Vtmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	for i = 1:time_index
		Btmp = B_list[:,:,i] * Utmp * Diagonal(Dtmp)
		Budv = svd_wrap(Btmp)
		Utmp = Budv.U
		Dtmp = Budv.D
		Vtmp = Budv.V * Vtmp
	end
	SVD_Store(Utmp, Dtmp, Vtmp)
end

function B_β_τ(time_index::Int, B_list::Array{Mat_Type,3}, MatDim::Int, N_time_slice::Int)::SVD_Store
	Btmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Utmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Dtmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Vtmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	for i = N_time_slice:(-1):(time_index + 1)
		Btmp = Diagonal(Dtmp) * Vtmp * B_list[:,:,i]
		Budv = svd_wrap(Btmp)
		Utmp = Utmp * Budv.U
		Dtmp = Budv.D
		Vtmp = Budv.V
	end
	SVD_Store(Utmp, Dtmp, Vtmp)
end


"""
## The propagation of B Matrix multiplication

As mentioned in `data.jl`, we have applied a mechanism to store the B matrix multiplication, 
which could save lots of time on B matrices multiplication. Thus we must update such storage 
after we update auxiliary field and then B matrix.

"""
function B_τ_0_prop!(B_mat::Matrix{Mat_Type}, tmp_mat::Matrix{Mat_Type}, tmp_udv::SVD_Store, dest::SVD_Store)
	# tmp_mat = B_mat * dest.U * Diagonal(dest.D)
	mul!(tmp_udv.U, B_mat, dest.U)
	rmul!(tmp_udv.U, Diagonal(dest.D))
	copyto!(tmp_mat, tmp_udv.U)
	svd_wrap!(tmp_mat, tmp_udv) # U, D, V = SVD( B_mat * dest.U * Diagonal(dest.D) )
	copyto!(dest.U, tmp_udv.U)
	copyto!(dest.D, tmp_udv.D)
	# ↓ dest.V = tmp_udv.V * dest.V
	copyto!(tmp_mat, dest.V)
	mul!(dest.V, tmp_udv.V, tmp_mat)
end

function B_β_τ_prop!(B_mat::AbstractArray{Mat_Type}, tmp_mat::Matrix{Mat_Type}, tmp_udv::SVD_Store, dest::SVD_Store)
	# tmp_mat = Diagonal(dest.D) * dest.V * B_mat
	mul!(tmp_udv.V, dest.V, B_mat)
	lmul!(Diagonal(dest.D), tmp_udv.V)
	copyto!(tmp_mat, tmp_udv.V)
	svd_wrap!(tmp_mat, tmp_udv)
	# dest.U = dest.U * tmp_udv.U
	copyto!(tmp_mat, dest.U)
	mul!(dest.U, tmp_mat, tmp_udv.U)
	copyto!(dest.D, tmp_udv.D)
	copyto!(dest.V, tmp_udv.V)
end

function fill_b_udv_store!(B_list::Array{Mat_Type,3}, udv_store::Vector{SVD_Store{Mat_Type}}, N_ns_int::Int, N_ns::Int, 
						tmp_mat::Matrix{Mat_Type}, tmp_udv::SVD_Store, ::Val{:β_τ})
	
	for i = N_ns:(-1):1
		copyto!(udv_store[i], udv_store[i + 1])
		start = i * N_ns_int
		stop = (i - 1) * N_ns_int + 1
		for j = start:(-1):stop
			@views B_β_τ_prop!(B_list[:,:,j], tmp_mat, tmp_udv, udv_store[i])
		end
	end
end

function sweep!(G_up::Matrix{Mat_Type}, G_dn::Matrix{Mat_Type}, 
				B_up_l::Array{Mat_Type,3}, B_dn_l::Array{Mat_Type,3}, tmp::temporary, pst::persistent,
				sl::Square_Lattice,	MatDim::Int, N_ns_int::Int, N_time_slice::Int, N_ns::Int, obser_switch::Bool)
	
	measure_time_index = rand(1:N_time_slice)

	copyto!(pst.B_τ_0_up_udv[2], pst.B_τ_0_up_udv[1])
	copyto!(pst.B_τ_0_dn_udv[2], pst.B_τ_0_dn_udv[1])

	for time_index = 1:N_time_slice
		last_time_index = time_index - 1
		if last_time_index == 0
			last_time_index = N_time_slice
		end
		# According to the B matrix storage scheme, when sweep up, we store the multiplication into the next.
		cur_udv_index = udv_index(time_index, N_ns_int)
		next_udv_index = cur_udv_index + 1

		if time_index % N_ns_int == 0

			# Here we are supposed to calculate Green's function at current time slice firstly.
			# ATTENTION !!!
			# After auxiliary field update, the B matrix also changes. So we MUST NOT multiply the 
			# old B Matrix into the storage.
	
			copyto!(tmp.udv_up, pst.B_τ_0_up_udv[next_udv_index])
			copyto!(tmp.udv_dn, pst.B_τ_0_dn_udv[next_udv_index])

			B_τ_0_prop!(B_up_l[:,:,time_index], tmp.mat, tmp.udv, tmp.udv_up)
			B_τ_0_prop!(B_dn_l[:,:,time_index], tmp.mat, tmp.udv, tmp.udv_dn)

			B_β_τ_up = pst.B_β_τ_up_udv[next_udv_index]
			B_β_τ_dn = pst.B_β_τ_dn_udv[next_udv_index]

			# G_up_tmp = B_up_l[:,:,time_index] * G_up * B_up_inv(sl.aux_field[:, time_index], sl.exp_mT)
			# G_dn_tmp = B_dn_l[:,:,time_index] * G_dn * B_dn_inv(sl.aux_field[:, time_index], sl.exp_mT)

			G_up[:,:] = G_σ_τ_τ_calc(tmp.udv_up, B_β_τ_up, MatDim)
			G_dn[:,:] = G_σ_τ_τ_calc(tmp.udv_dn, B_β_τ_dn, MatDim)

			# G_up_err_max = maximum(G_up - G_up_tmp)
			# G_dn_err_max = maximum(G_dn - G_dn_tmp)

			# println("↑ ", G_up_err_max, " ",G_dn_err_max)
		else
			B_up_inv!(sl.aux_field[:, time_index], sl.exp_mT, tmp.exp_V, tmp.b_inv)
			# G_up[:,:] = B_up_l[:,:,time_index] * G_up * tmp.b_inv
			mul!(tmp.mat, G_up, tmp.b_inv)
			mul!(G_up, view(B_up_l, :, :, time_index), tmp.mat)
			# ------------------------------------------------------
			B_dn_inv!(sl.aux_field[:, time_index], sl.exp_mT, tmp.exp_V, tmp.b_inv)
			# G_dn[:,:] = B_dn_l[:,:,time_index] * G_dn * tmp.b_inv
			mul!(tmp.mat, G_dn, tmp.b_inv)
			mul!(G_dn, view(B_dn_l, :, :, time_index), tmp.mat)
		end

		update!(time_index, G_up, G_dn, B_up_l, B_dn_l, sl, tmp.exp_V, MatDim)
		B_τ_0_prop!(B_up_l[:,:,time_index], tmp.mat, tmp.udv, pst.B_τ_0_up_udv[next_udv_index])
		B_τ_0_prop!(B_dn_l[:,:,time_index], tmp.mat, tmp.udv, pst.B_τ_0_dn_udv[next_udv_index])

		if time_index % N_ns_int == 0 && cur_udv_index != N_ns
			copyto!(pst.B_τ_0_up_udv[next_udv_index + 1], pst.B_τ_0_up_udv[next_udv_index])
			copyto!(pst.B_τ_0_dn_udv[next_udv_index + 1], pst.B_τ_0_dn_udv[next_udv_index])
		end

		if obser_switch && time_index == measure_time_index
			sampling(G_up, G_dn, sl)
		end
	end

	# Since we have updated the Green's Function at time_index = N_time_slice while sweep up. 
	# We just start at time_index = N_time_slice - 1 when sweep down.
	# But we must not forget to propagate B.
	copyto!(pst.B_β_τ_up_udv[N_ns], pst.B_β_τ_up_udv[N_ns + 1])
	copyto!(pst.B_β_τ_dn_udv[N_ns], pst.B_β_τ_dn_udv[N_ns + 1])
	B_β_τ_prop!(B_up_l[:,:,N_time_slice], tmp.mat, tmp.udv, pst.B_β_τ_up_udv[N_ns])
	B_β_τ_prop!(B_dn_l[:,:,N_time_slice], tmp.mat, tmp.udv, pst.B_β_τ_dn_udv[N_ns])

	for time_index = (N_time_slice - 1):-1:1
		last_time_index = time_index + 1

		cur_udv_index = udv_index(time_index, N_ns_int)

		if time_index % N_ns_int == 0
			copyto!(pst.B_β_τ_up_udv[cur_udv_index], pst.B_β_τ_up_udv[cur_udv_index + 1])
			copyto!(pst.B_β_τ_dn_udv[cur_udv_index], pst.B_β_τ_dn_udv[cur_udv_index + 1])

			B_τ_0_up = pst.B_τ_0_up_udv[cur_udv_index + 1]
			B_τ_0_dn = pst.B_τ_0_dn_udv[cur_udv_index + 1]

			@views G_up[:,:] = G_σ_τ_τ_calc(B_τ_0_up, pst.B_β_τ_up_udv[cur_udv_index + 1], MatDim)
			@views G_dn[:,:] = G_σ_τ_τ_calc(B_τ_0_dn, pst.B_β_τ_dn_udv[cur_udv_index + 1], MatDim)
		else
			B_up_inv!(sl.aux_field[:,last_time_index], sl.exp_mT, tmp.exp_V, tmp.b_inv)
			# G_up[:,:] = tmp.b_inv * G_up * B_up_l[:,:,last_time_index]
			mul!(tmp.mat, G_up, B_up_l[:,:,last_time_index])
			mul!(G_up, tmp.b_inv, tmp.mat)
			# ------------------------------------------------------
			B_dn_inv!(sl.aux_field[:,last_time_index], sl.exp_mT, tmp.exp_V, tmp.b_inv)
			# G_dn[:,:] = tmp.b_inv * G_dn * B_dn_l[:,:,last_time_index]
			mul!(tmp.mat, G_dn, B_dn_l[:,:,last_time_index])
			mul!(G_dn, tmp.b_inv, tmp.mat)
		end

		update!(time_index, G_up, G_dn, B_up_l, B_dn_l, sl, tmp.exp_V, MatDim)

		B_β_τ_prop!(B_up_l[:,:,time_index], tmp.mat, tmp.udv, pst.B_β_τ_up_udv[cur_udv_index])
		B_β_τ_prop!(B_dn_l[:,:,time_index], tmp.mat, tmp.udv, pst.B_β_τ_dn_udv[cur_udv_index])
	end
end

function update!(time_index::Int, G_up::Matrix{Mat_Type}, G_dn::Matrix{Mat_Type}, 
	B_up_l::Array{Mat_Type,3}, B_dn_l::Array{Mat_Type,3}, sl::Square_Lattice, exp_V_tmp::Vector{Mat_Type}, MatDim::Int)

	delta_V_up = 0.0
	delta_V_dn = 0.0
	for i = 1:MatDim
		if sl.aux_field[i, time_index] == 1
			delta_V_up = Param.exp_mα / Param.exp_α - 1
			delta_V_dn = Param.exp_α / Param.exp_mα - 1
		else
			delta_V_up = Param.exp_α / Param.exp_mα - 1
			delta_V_dn = Param.exp_mα / Param.exp_α - 1
		end

		R_up = 1 + delta_V_up * (1 - G_up[i,i])
		R_dn = 1 + delta_V_dn * (1 - G_dn[i,i])

		R = R_up * R_dn
		@assert R > 0 "R should larger than zero, $time_index, $R_up, $R_dn"
		if rand() < R
			tmp = G_up[i,:]
			tmp = -tmp
			tmp[i] = tmp[i] + 1
			@views G_up[:,:] = G_up[:,:] - delta_V_up / R_up * G_up[:,i] * transpose(tmp)

			tmp = G_dn[i,:]
			tmp = -tmp
			tmp[i] = tmp[i] + 1
			@views G_dn[:,:] = G_dn[:,:] - delta_V_dn / R_dn * G_dn[:,i] * transpose(tmp)

			sl.aux_field[i,time_index] = -sl.aux_field[i,time_index]
		end
	end
	@views B_up!(sl.aux_field[:,time_index], sl.exp_T, exp_V_tmp, B_up_l[:,:,time_index])
	@views B_dn!(sl.aux_field[:,time_index], sl.exp_T, exp_V_tmp, B_dn_l[:,:,time_index])
end

function G_σ_τ_τ_calc(R::SVD_Store, L::SVD_Store, N_dim::Int)
	U_R, D_R, V_R = R.U, Diagonal(R.D), R.V
	V_L, D_L, U_L = L.U, Diagonal(L.D), L.V

	D_R_max = zeros(N_dim)
	D_R_min = zeros(N_dim)
	D_L_max = zeros(N_dim)
	D_L_min = zeros(N_dim)

	for i = 1:N_dim
		if real(D_R[i,i]) > 1
			D_R_max[i] = D_R[i,i]
			D_R_min[i] = 1
		else
			D_R_min[i] = D_R[i,i]
			D_R_max[i] = 1
		end
		if real(D_L[i,i]) > 1
			D_L_max[i] = D_L[i,i]
			D_L_min[i] = 1
		else
			D_L_min[i] = D_L[i,i]
			D_L_max[i] = 1
		end
	end

	D_R_max_inv = LinearAlgebra.Diagonal(1 ./ D_R_max)
	D_L_max_inv = LinearAlgebra.Diagonal(1 ./ D_L_max)

	G_σ_τ_τ = inv(U_L) * D_L_max_inv *
			inv(D_R_max_inv * inv(U_L * U_R) * D_L_max_inv +
				LinearAlgebra.Diagonal(D_R_min) * V_R * V_L * LinearAlgebra.Diagonal(D_L_min)) *
			D_R_max_inv * inv(U_R)
	G_σ_τ_τ
end

end