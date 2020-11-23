include("hubbard.jl")
include("svd.jl")
include("data.jl")
include("obser.jl")
include("core.jl")

function main()
	hm = Hubbard_Model
	p = Hubbard_Model.Param

	aux_field = Hubbard_Model.gen_auxf()
	nlist = Hubbard_Model.gen_nlist()
	T = Hubbard_Model.T(nlist)

	# The following two lines could be slow when size is large, since we haven't implement checkerboard decomposition.
	exp_T = exp(p.Δτ * T)
	exp_mT = exp(-p.Δτ * T)

	sl = Hubbard_Model.Square_Lattice(aux_field, nlist, T, exp_T, exp_mT)

	pst_data = Data.persistent(hm.Mat_Type, p.N_ns, p.MatDim)
	tmp_data = Data.temporary(hm.Mat_Type, p.MatDim)

	B_up_l, B_dn_l = CoreM.init_B_mat_list!(sl.aux_field, sl.exp_T, tmp_data, p.MatDim, p.N_time_slice)

	SVDM.init_b_udv_store!(pst_data.B_β_τ_up_udv, p.N_ns, p.MatDim, Val(:β_τ))
	SVDM.init_b_udv_store!(pst_data.B_β_τ_dn_udv, p.N_ns, p.MatDim, Val(:β_τ))

	SVDM.init_b_udv_store!(pst_data.B_τ_0_up_udv, p.N_ns, p.MatDim, Val(:τ_0))
	SVDM.init_b_udv_store!(pst_data.B_τ_0_dn_udv, p.N_ns, p.MatDim, Val(:τ_0))

	CoreM.fill_b_udv_store!(B_up_l, pst_data.B_β_τ_up_udv, p.N_ns_int, p.N_ns, tmp_data.mat, tmp_data.udv, Val(:β_τ))
	CoreM.fill_b_udv_store!(B_dn_l, pst_data.B_β_τ_dn_udv, p.N_ns_int, p.N_ns, tmp_data.mat, tmp_data.udv, Val(:β_τ))

	G_up = CoreM.G_σ_τ_τ_calc(pst_data.B_τ_0_up_udv[1], pst_data.B_β_τ_up_udv[1], p.MatDim)
	G_dn = CoreM.G_σ_τ_τ_calc(pst_data.B_τ_0_dn_udv[1], pst_data.B_β_τ_dn_udv[1], p.MatDim)

	println("initialization done")

	for i = 1:p.N_warmup
		print(i, "...")
		CoreM.sweep!(G_up, G_dn, B_up_l, B_dn_l, tmp_data, pst_data, sl, 
				p.MatDim, p.N_ns_int, p.N_time_slice, p.N_ns, false)
	end
	println()
	for i = 1:p.N_bin
		println("bin=", i)
		@time for j = 1:p.N_sweep
			print(j, "...")
			CoreM.sweep!(G_up, G_dn, B_up_l, B_dn_l, tmp_data, pst_data, sl, 
				p.MatDim, p.N_ns_int, p.N_time_slice, p.N_ns, true)
		end
		Obser.bin_store(i)
		println()
	end
end