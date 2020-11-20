module CoreM

using LinearAlgebra
using ..SVDM: SVD_Store, svd_wrap
using ..Hubbard_Model: expV, Param, Square_Lattice
using ..Obser: sampling

function B_up(auxF::Vector{Int}, exp_T::Matrix{Float64})
	expV(1, auxF) * exp_T
end

function B_dn(auxF::Vector{Int}, exp_T::Matrix{Float64})
	expV(-1, auxF) * exp_T
end

function B_up_inv(auxF::Vector{Int}, exp_mT::Matrix{Float64})
	exp_mT * expV(-1, auxF)
end

function B_dn_inv(auxF::Vector{Int}, exp_mT::Matrix{Float64})
	exp_mT * expV(1, auxF)
end

function init_B_mat_list(auxf::Matrix{Int}, exp_T::Matrix{Float64}, MatDim::Int, N_time_slice::Int)
	B_up_list = Array{Float64,3}(undef, MatDim, MatDim, N_time_slice)
	B_dn_list = Array{Float64,3}(undef, MatDim, MatDim, N_time_slice)
	for i = 1:N_time_slice
		B_up_list[:,:,i] = B_up(auxf[:,i], exp_T)
		B_dn_list[:,:,i] = B_dn(auxf[:,i], exp_T)
	end
	return B_up_list, B_dn_list
end

function B_τ_0(time_index::Int, B_list::Array{Float64,3}, MatDim::Int)::SVD_Store
	Btmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Utmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Dtmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Vtmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	for i = 1:time_index
		Btmp = B_list[:,:,i] * Utmp * Dtmp
		Budv = svd_wrap(Btmp)
		Utmp = Budv.U
		Dtmp = Budv.D
		Vtmp = Budv.V * Vtmp
	end
	SVD_Store(Utmp, Dtmp, Vtmp)
end

function B_β_τ(time_index::Int, B_list::Array{Float64,3}, MatDim::Int, N_time_slice::Int)::SVD_Store
	Btmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Utmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Dtmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	Vtmp = Matrix(LinearAlgebra.I, MatDim, MatDim)
	for i = N_time_slice:(-1):(time_index + 1)
		Btmp = Dtmp * Vtmp * B_list[:,:,i]
		Budv = svd_wrap(Btmp)
		Utmp = Utmp * Budv.U
		Dtmp = Budv.D
		Vtmp = Budv.V
	end
	SVD_Store(Utmp, Dtmp, Vtmp)
end

function sweep!(G_up::Matrix{Float64}, G_dn::Matrix{Float64}, 
				B_up_l::Array{Float64,3}, B_dn_l::Array{Float64,3},
				sl::Square_Lattice,	MatDim::Int, N_ns_int::Int, N_time_slice::Int, obser_switch::Bool)
	
	measure_time_index = rand(1:N_time_slice)
	for time_index = 1:N_time_slice
		last_time_index = time_index - 1
		if last_time_index == 0
			last_time_index = N_time_slice
		end
		
		if time_index % N_ns_int == 0
			B_τ_0_up = B_τ_0(time_index, B_up_l, MatDim)
			B_τ_0_dn = B_τ_0(time_index, B_dn_l, MatDim)

			B_β_τ_up = B_β_τ(time_index, B_up_l, MatDim, N_time_slice)
			B_β_τ_dn = B_β_τ(time_index, B_dn_l, MatDim, N_time_slice)

			# G_up_tmp = B_up_l[:,:,time_index] * G_up * B_up_inv(sl.aux_field[:, time_index], sl.exp_mT)
			# G_dn_tmp = B_dn_l[:,:,time_index] * G_dn * B_dn_inv(sl.aux_field[:, time_index], sl.exp_mT)

			G_up[:,:] = G_σ_τ_τ_calc(B_τ_0_up, B_β_τ_up, MatDim)
			G_dn[:,:] = G_σ_τ_τ_calc(B_τ_0_dn, B_β_τ_dn, MatDim)

			# G_up_err_max = maximum(G_up - G_up_tmp)
			# G_dn_err_max = maximum(G_dn - G_dn_tmp)

			# println("↑ ", G_up_err_max, " ",G_dn_err_max)
		else
			G_up[:,:] = B_up_l[:,:,time_index] * G_up * B_up_inv(sl.aux_field[:, time_index], sl.exp_mT)
			G_dn[:,:] = B_dn_l[:,:,time_index] * G_dn * B_dn_inv(sl.aux_field[:, time_index], sl.exp_mT)
		end

		update!(time_index, G_up, G_dn, B_up_l, B_dn_l, sl, MatDim)

		if obser_switch && time_index == measure_time_index
			sampling(G_up, G_dn, sl)
		end
	end

	for time_index = (N_time_slice - 1):-1:1
		last_time_index = time_index + 1
		if time_index % N_ns_int == 0
			B_τ_0_up = B_τ_0(time_index, B_up_l, MatDim)
			B_τ_0_dn = B_τ_0(time_index, B_dn_l, MatDim)

			B_β_τ_up = B_β_τ(time_index, B_up_l, MatDim, N_time_slice)
			B_β_τ_dn = B_β_τ(time_index, B_dn_l, MatDim, N_time_slice)

			G_up[:,:] = G_σ_τ_τ_calc(B_τ_0_up, B_β_τ_up, MatDim)
			G_dn[:,:] = G_σ_τ_τ_calc(B_τ_0_dn, B_β_τ_dn, MatDim)
		else
			G_up[:,:] = B_up_inv(sl.aux_field[:,last_time_index], sl.exp_mT) * G_up * B_up_l[:,:,last_time_index]
			G_dn[:,:] = B_dn_inv(sl.aux_field[:,last_time_index], sl.exp_mT) * G_dn * B_dn_l[:,:,last_time_index]
		end

		update!(time_index, G_up, G_dn, B_up_l, B_dn_l, sl, MatDim)
	end
end

function update!(time_index::Int, G_up::Matrix{Float64}, G_dn::Matrix{Float64}, 
	B_up_l::Array{Float64,3}, B_dn_l::Array{Float64,3}, sl::Square_Lattice, MatDim::Int)
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
		@assert R > 0 "R should larger than zero"
		if rand() < R
			tmp = G_up[i,:]
			tmp = -tmp
			tmp[i] = tmp[i] + 1
			G_up[:,:] = G_up[:,:] - delta_V_up / R_up * G_up[:,i] * transpose(tmp)

			tmp = G_dn[i,:]
			tmp = -tmp
			tmp[i] = tmp[i] + 1
			G_dn[:,:] = G_dn[:,:] - delta_V_dn / R_dn * G_dn[:,i] * transpose(tmp)

			sl.aux_field[i,time_index] = -sl.aux_field[i,time_index]
		end
	end
	B_up_l[:,:,time_index] = B_up(sl.aux_field[:,time_index], sl.exp_T)
	B_dn_l[:,:,time_index] = B_dn(sl.aux_field[:,time_index], sl.exp_T)
end

function G_σ_τ_τ_calc(R::SVD_Store, L::SVD_Store, N_dim::Int)
	U_R, D_R, V_R = R.U, R.D, R.V
	V_L, D_L, U_L = L.U, L.D, L.V

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