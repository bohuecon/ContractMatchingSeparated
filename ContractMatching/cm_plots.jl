
function plot_equ(;para = para_default)

    @unpack vec_i, vec_e, num_i = para

    # plot internal contracts

    mat_Mu = load("cm.jld", "mat_Mu")
    contour(vec_i, vec_e, mat_Mu', xlabel = "i, the firm type", ylabel = "e, the executive type", fill = true, c = :OrRd_9, rev = true, size = (800, 600), title = "Matchings")
    Plots.savefig("./figures/internalContractMu.pdf")

    mat_Mu_bm = load("cm.jld", "mat_Mu_bm")
    contour(vec_i, vec_e, mat_Mu_bm', xlabel = "i, the firm type", ylabel = "e, the executive type", fill = true, c = :OrRd_9, rev = true, size = (800, 600), title = "Matchings (BM)")
    Plots.savefig("./figures/internalContractMuBm.pdf")

    # mat_cStar = load("internal_candi.jld", "mat_cStar")
    # # # mat_cstar[mat_cstar.<0.0] .= 0.0
    # # # surface(vec_i, vec_e, mat_cstar)
    # # # heatmap(vec_i, vec_e, mat_cstar, aspect_ratio = 1)
    # contour(vec_i, vec_e, mat_cStar', xlabel = L"i", ylabel = L"e", fill = true, c = :OrRd_9, rev = true, size = (800, 600), title = L"c^\ast")
    # Plots.savefig("./figures/internalContractC.pdf")

    # mat_dummiesStar = load("internal_candi.jld", "mat_dummiesStar")
    # mat_d1Star = [dummies[1] for dummies in mat_dummiesStar]
    # mat_d2Star = [dummies[2] for dummies in mat_dummiesStar]
    # # mat_d3Star = [dummies[3] for dummies in mat_dummiesStar]

    p1 = contour(vec_i, vec_e, mat_d1Star', xlabel = L"i", ylabel = L"e", fill = true, c = :OrRd_9, rev = true, title = L"d_1",size = (800, 600))
    # Plots.savefig("./figures/internalContractd1.pdf")

    # p2 = contour(vec_i, vec_e, mat_d2Star', xlabel = L"i", ylabel = L"e", fill = true, c = :OrRd_9, rev = true, title = L"d_2", size = (800, 600))
    # Plots.savefig("./figures/internalContractd2.pdf")

    # # p3 = contour(vec_i, vec_e, mat_d3Star', xlabel = L"i", ylabel = L"e", fill = true, c = :OrRd_9, rev = true, title = L"d_3", size = (800, 600))
    # # Plots.savefig("./figures/internalContractd3.pdf")

    # # plot(p1, p2, p3, layout=(2,2), size = (1600, 500))
    # # Plots.savefig("internalContractDummies.pdf")

    # # plot external contracts

    # arr_Mu = load("internal_candi.jld", "arr_Mu");
    # arr_cStar = load("internal_candi.jld", "arr_cStar");

    # vec_iota_ind = [1, Int(floor(num_i*(3/5))), Int(floor(num_i*(4/5))), num_i]

    # # vec_figures = []

    # for iota_ind in vec_iota_ind

    #     iota = round(vec_i[iota_ind], digits=1)
    #     mat_Mu_iota = arr_Mu[:, :, iota_ind]
    #     mat_cStar_iota = arr_cStar[:, :, iota_ind]
        
    #     p = contour(vec_i, vec_e, mat_Mu_iota', xlabel = L"i", ylabel = L"e", fill = true, c = :OrRd_9, rev = true, title = "Matchings (iota = $iota)", size = (800, 600))
    #     # push!(vec_figures, p)
    #     filename = "./figures/externalContractMu$iota_ind.pdf"
    #     Plots.savefig(filename)

    #     p = contour(vec_i, vec_e, mat_cStar_iota', xlabel = L"i", ylabel = L"e", fill = true, c = :OrRd_9, rev = true, title = "c (iota = $iota)", size = (800, 600))
    #     # push!(vec_figures, p)
    #     filename = "./figures/externalContractcStar$iota_ind.pdf"
    #     Plots.savefig(filename)
    # end

end 




