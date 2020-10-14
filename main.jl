include("hubbard.jl")
include("svd.jl")
include("obser.jl")
include("core.jl")

function main()
	hm = Hubbard_Model
	p = Hubbard_Model.Param

	aux_field = Hubbard_Model.gen_auxf()
	nlist = Hubbard_Model.gen_nlist()
	T = Hubbard_Model.T(nlist)

	# This could be slow, since no checkerboard decomposition.
	exp_T = exp(p.Δτ * T)
	exp_mT = exp(-p.Δτ * T)

	sl = Hubbard_Model.Square_Lattice(aux_field, nlist, T, exp_T, exp_mT)

	B_up_l, B_dn_l = CoreM.init_B_mat_list(sl.aux_field, sl.exp_mT, p.MatDim, p.N_time_slice)

	B_τ_0_up = CoreM.B_τ_0(p.N_time_slice, B_up_l, p.MatDim)
	B_τ_0_dn = CoreM.B_τ_0(p.N_time_slice, B_dn_l, p.MatDim)

	B_β_τ_up = CoreM.B_β_τ(0, B_up_l, p.MatDim, p.N_time_slice)
	B_β_τ_dn = CoreM.B_β_τ(0, B_dn_l, p.MatDim, p.N_time_slice)

	G_up = CoreM.G_σ_τ_τ_calc(B_τ_0_up, B_β_τ_up, p.MatDim)
	G_dn = CoreM.G_σ_τ_τ_calc(B_τ_0_dn, B_β_τ_dn, p.MatDim)

	println("initialization done")

	for i=1:p.N_warmup
		print(i,"...")
		CoreM.sweep!(G_up, G_dn, B_up_l, B_dn_l, sl, p.MatDim, p.N_ns_int, p.N_time_slice, false)
	end
	println()
	for i=1:p.N_bin
		println("bin=",i)
		for j=1:p.N_sweep
			print(j,"...")
			CoreM.sweep!(G_up,G_dn, B_up_l, B_dn_l, sl, p.MatDim, p.N_ns_int, p.N_time_slice, true)
		end
		Obser.bin_store(i)
		println()
	end
end