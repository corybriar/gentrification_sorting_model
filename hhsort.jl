function household_sort(para, space, firms, prices, rents, wages, γ, n, εh, εw, nloc)
    @unpack αU, αS, ζ, η, τ, T, IU, IS, M, S, σw, σh = para
    E = [1, 2]
    I = [IU, IS]
    α = [αU, αS]
    # Construct Pl
    Pl = Pℓ(para, γ, firms, prices, nloc)
    # Blank array to hold max for each city
    maxm = zeros(IU + IS, 4, M)
    # Blank matrix to hold π^e_s(ℓ'|ℓ)
    V = []
    sumV = []
    for m in 1:M
        Vesm = zeros(2, nloc[m],2,S)
            # Vesm[1,ℓ,e,s] = utility
            # Vesm[2,ℓ,e,s] = income
        sumVesm = zeros(2,S)
        for e in E, s in 1:S
            VW = (1/σw) .* (
                log(n[e,s,m]) .+ log.(wages[m][:,e,s]')
                .+ η.*log.(T .- n[e,s,m] .- γ[m])
                )
            πm = exp.(VW) .* ((ones(1,nloc[m])*(exp.(VW))').^(-1))
            # fill Vesm with utilities for each location under sector s and education e
            Vesm[1,:,e,s] = transpose((1/σh).*(diag(πm .* transpose(VW)) .-
                α[e].*log.(rents[m]) .- (1-α[e]).*log.(Pl[m])))
            # Back out incomes for those in ℓ ∈ Lm
            Vesm[2,:,e,s] = transpose(πm)*wages[m][:,e,s]
            # sum up utilies across locations
            sumVesm[e,s] = sum(exp.(Vesm[1,:,e,s]))
        end # e-s loop
        push!(V, Vesm)
        push!(sumV,sumVesm)
    end # m loop

    Vdenom =  sum(sumV)
    pop = []
    HX = []
    for m in 1:M
        popm = zeros(nloc[m],2,S)
        HXm = zeros(2,nloc[m], 2, S)
        for e in 1:2, s in 1:S
            popm[:,e,s] = I[e] .* exp.(V[m][1,:,e,s]) .* (sum(Vdenom[e,:])).^(-1)
            HXm[1,:,e,s] = I[e]*α[e] .* popm[:,e,s] .* V[m][2,:,e,s] ./ rents[m][:]
            HXm[2,:,e,s] = I[e]*(1-α[e]) .* popm[:,e,s] .* V[m][2,:,e,s] ./ Pl[m][:]
        end # e loop
        push!(pop, popm)
        push!(HX, HXm)
    end # m loop




    for m in 1:M
        # Blank matrix to hold value of ℓ,s combo in m
        πm = zeros(nloc[m],nloc[m],2,S)
        Vsm = zeros(IU + IS, nloc[m], S)
        for e in E
            for s in 1:S
                # construct utilities of working in l' given living in l
                VW = (1/σw) .* (
                    log(n[e,s,m]) .+ log.(wages[m][:,e,s]')
                    .+ η.*log.(T .- n[e,s,m] .- γ[m])
                    )
                # post residential sort work probabilities
                πm[:,:,e,s] = exp.(VW) .* ((ones(1,nloc[m])*(exp.(VW))').^(-1))

                # fill Vsm with utilities for each location under sector s
                if e == 1
                    Vsm[1:IU,:,s] = transpose((1/σh).*(diag(πm[:,:,e,s] .* transpose(VW)) .-
                        α[e].*log.(rents[m]) .- (1-α[e]).*log.(Pl[m]))) .+ εh[m][1:IU,:]
                else
                    Vsm[(IU+1):(IS + IU),:,s] = transpose((1/σh).*(diag(πm[:,:,e,s] .* transpose(VW)) .-
                        α[e].*rents[m] .- (1-α[e]).*Pl[m])) .+ εh[m][(IU+1):(IS + IU),:]
                end
            end # s loop

            if e == 1
                # find maximizing choice ℓ in city m,s
                maxm[1:IU,1,m]  = findmax(Vsm[1:IU,:,:], dims = (2,3))[1]
                maxm[1:IU,4,m]  .= 1
                for i in 1:I[e]
                    maxm[i,2,m] = findmax(Vsm, dims = (2,3))[2][i][2]
                    maxm[i,3,m] = findmax(Vsm, dims = (2,3))[2][i][3]
                end # i loop
            else
                # find maximizing choice ℓ in city m,s
                maxm[(IU+1):(IS + IU),1,m]  = findmax(Vsm[(IU+1):(IS + IU),:,:], dims = (2,3))[1]
                maxm[(IU+1):(IS + IU),4,m] .= 2
                for i in (I[e-1]):(I[e] + I[e-1])
                    maxm[i,2,m] = findmax(Vsm, dims = (2,3))[2][i][2]
                    maxm[i,3,m] = findmax(Vsm, dims = (2,3))[2][i][3]
                end # i loop
            end # if statement
        end # e loop
        # add to πlpgl array for use inb calculating incomes
        push!(πlpgl,πm)
    end # m loop
    # Find utility maximizing city for each agent
    Mmax = zeros(IS + IU)
    for i in 1:(IS + IU)
        Mmax[i] = findmax(maxm[i,1,:])[2]
    end

    Mmax = Int.(Mmax)
    # Vector of best choices and income, [ℓ,s,e,m]
    Vmax = [maxm[1,2:4,Mmax]; Mmax']'

    # Compute household income's to infer demand for goods
    HX = []
    # HX[m] = [ℓ,ω,H,X]
    for m in 1:M
        # HXm = [ℓ,ωi,Hi,Xi]
        HXm = zeros(0,4)
        for e in E
            for s in 1:S
                # Filter down to only individuals with (s,e,m)
                vmax = Vmax[Vmax[:,2] .== s,:]
                vmax = vmax[vmax[:,3] .== e,:]
                vmax = vmax[vmax[:,4] .== m,:]
                vmax = Int.(vmax)
                # Calculate expected wages
                ωmse = transpose(πlpgl[m][:,vmax[:,1],e,s])*wages[m][:,e,s]
                HXmse = hcat(vmax[:,1],
                    ωmse,
                    ωmse.*α[vmax[:,3]]./rents[m][vmax[:,1]] ,
                    ωmse.*(1 .-α[vmax[:,3]])./Pl[m][vmax[:,1]])
                HXm = [HXm ; HXmse]
            end # s loop
        end # e loop
        push!(HX,HXm)
    end # m loop

    return Vmax, HX
end
