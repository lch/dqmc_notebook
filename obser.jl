module Obser

using ..Hubbard_Model: Param, Neighbor, Square_Lattice
using LinearAlgebra

kinetic_sum = 0.0
dbl_occu_sum = 0.0

kinetic_store = Vector{Float64}(undef, Param.N_bin)
dbl_occu_store = Vector{Float64}(undef, Param.N_bin)

function kinetic(gttupc::Matrix{Float64},gttdnc::Matrix{Float64}, nlist::Vector{Neighbor})
	k = 0.0
	for i=1:Param.N_pc
		up = nlist[i].up
		down = nlist[i].down
		left = nlist[i].left
		right = nlist[i].right

		k += gttupc[i,up] + gttupc[up,i]
		k += gttdnc[i,up] + gttdnc[up,i]

		k += gttupc[i,down] + gttupc[down,i]
		k += gttdnc[i,down] + gttdnc[down,i]

		k += gttupc[i,left] + gttupc[left,i]
		k += gttdnc[i,left] + gttdnc[left,i]

		k += gttupc[i,right] + gttupc[right,i]
		k += gttdnc[i,right] + gttdnc[right,i]
	end
	-Param.t * k / (2 * Param.MatDim)
end

function dbl_occu(gttupc::Matrix{Float64}, gttdnc::Matrix{Float64})
	doc = 0.0
	for i=1:Param.MatDim
		doc += gttupc[i,i] * gttdnc[i,i]
	end
	doc / Param.MatDim
end

function sampling(G_up::Matrix{Float64}, G_dn::Matrix{Float64}, sl::Square_Lattice)
	eye = Matrix(LinearAlgebra.I, Param.MatDim, Param.MatDim)
	G_upc = eye - transpose(G_up)
	G_dnc = eye - transpose(G_dn)
	global kinetic_sum += kinetic(G_upc,G_dnc,sl.nlist)
	global dbl_occu_sum += dbl_occu(G_upc, G_dnc)
end

function bin_store(bin::Int)
	global kinetic_store[bin] = kinetic_sum / Param.N_sweep
	global kinetic_sum = 0.0
	global dbl_occu_store[bin] = dbl_occu_sum / Param.N_sweep
	global dbl_occu_sum = 0.0
end

end