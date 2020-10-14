### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ 0a5c6200-0d0b-11eb-15f7-2f2818b0e133
module Hubbard_Model

import LinearAlgebra.Diagonal

module Param
# Below is what users set
Lx = 4
Ly = 4
t = 1.0
U = 1.0
β = 4
Δτ = 0.05
μ = 0

N_warmup = 100
N_sweep = 100
N_bin = 5
N_ns_int = 10

# Below is calculated by program
α = acosh(exp(0.5 * Δτ * U))
exp_α = exp(α)
exp_mα = exp(-α)
N_pc = Lx * Ly
N_bond = N_pc * 2
N_time_slice =  Int(β / Δτ)
N_ns = N_time_slice ÷ N_ns_int
MatDim = N_pc

function set(var::Symbol, val)::Nothing
	@eval global $var = $val
	update()
	nothing
end

function update()::Nothing
	global α = acosh(exp(0.5 * Δτ * U))
	global exp_α = exp(α)
	global exp_mα = exp(-α)
	global N_pc = Lx * Ly
	global N_bond = N_pc * 2
	global N_time_slice =  Int(β / Δτ)
	global N_ns = N_time_slice ÷ N_ns_int
	global MatDim = N_pc
	nothing
end
end

mutable struct Neighbor
	up::Int
	down::Int
	left::Int
	right::Int
	function Neighbor(index::Int, Lx::Int, Ly::Int)
		ix, iy = cart_coord(index, Lx)
		up_y = iy - 1
		if up_y < 1
			up_y = Ly
		end
		up = linear_index(ix, up_y, Lx)
		down_y = iy + 1
		if down_y > Ly
			down_y = 1
		end
		down = linear_index(ix, down_y, Lx)
		left_x = ix - 1
		if left_x < 1
			left_x = Lx
		end
		left = linear_index(left_x, iy , Lx)
		right_x = ix + 1
		if right_x > Lx
			right_x = 1
		end
		right = linear_index(right_x, iy, Lx)
		new(up, down, left, right)
	end
end

mutable struct Square_Lattice
	aux_field::Matrix{Int}
	nlist::Vector{Neighbor}
	T::Matrix{Float64}
	exp_T::Matrix{Float64}
	exp_mT::Matrix{Float64}
end

p = Param
export p

function cart_coord(i::Int, Lx::Int)
	ix = i % Lx
	iy = i ÷ Lx + 1
	if i % Lx == 0
		iy -= 1
		ix = Lx
	end
	return ix, iy
end

function linear_index(ix::Int, iy::Int, Lx::Int)::Int
	(iy - 1) * Lx + ix
end

function gen_nlist()
	nlist = Vector{Neighbor}(undef, Param.N_pc)
	for i=1:Param.N_pc
		nlist[i] = Neighbor(i, Param.Lx, Param.Ly)
		# println("$i ", nlist[i])
	end
	nlist
end

function gen_auxf()
	rand([-1,1],Param.MatDim,Param.N_time_slice)
end

function T(nlist::Vector{Neighbor})
	T_mat = zeros(Float64, Param.MatDim, Param.MatDim)
	for i=1:Param.N_pc
		down = nlist[i].down
		right = nlist[i].right
		T_mat[i, down] = Param.t
		T_mat[down, i] = Param.t
		T_mat[i, right] = Param.t
		T_mat[right, i] = Param.t
	end
	T_mat
end

function expV(sigma::Int, auxF::Vector{Int})
	expV_mat = zeros(Float64, Param.MatDim, Param.MatDim)
	for i=1:Param.MatDim
		if auxF[i] == 1 * sigma
			expV_mat[i,i] = Param.exp_α
		else
			expV_mat[i,i] = Param.exp_mα
		end
	end
	expV_mat
end

end

# ╔═╡ Cell order:
# ╠═0a5c6200-0d0b-11eb-15f7-2f2818b0e133
