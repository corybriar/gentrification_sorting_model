using Pkg
using LinearAlgebra
using Plots
using Distributions
using Parameters

@with_kw struct parameters
    αU::Float64 = 0.5   # Utility parameter
    αS::Float64 = 0.6
    ζ::Float64 = -2     # Elasticity of substitution between goods
    η::Float64 = 1.5    # Elast. of utility to work/commute
    τ::Float64 = 1.3    # Travel cost parameter
    T::Float64 = 24
    IU::Int64 = 100     # Number of agents
    IS::Int64 = 100
    M::Int64 = 2        # Number of cities
    S::Int64  = 4       # Number of sectors
    σh::Float64 = 0.5   # sd of εh
    σw ::Float64= 0.5
    σϵ::Float64 = 0.5   # sd of ϵ
    φ::Float64 = 1
end

para = parameters()

ρ = [2 3 4 5]
ψ = [1.5, 1.5]
θ = [0.3, 0.4, 0.5, 0.6]
κ = 0.3
JS = [15,15,15,15]
n = 10 .*ones(2,4,2)

nloc = [15 15]
# create imaginary space to work with
function space_gen(para,nloc, sizemax)
    @unpack M = para
    space = []
    space_dist = Uniform(0,sizemax)
    for m in 1:M
        locs = rand(space_dist, nloc[m], 2)
        push!(space, locs)
    end # m loop
    return space
end

space = space_gen(para,nloc,1)

space =[]
push!(space, [0 0; 0 1])
push!(space, [0 0; 0 1])
nloc = [2, 2]

function γ_gen(para, space, nloc, cost)
    @unpack M = para
    com_costs = []
    # iterate by city
    for m in 1:M
        comcost_l = zeros(nloc[m],nloc[m])
        # iterate between each location
        for l1 in 1:nloc[m]
            for l2 in 1:nloc[m]
                comcost_l[l1,l2] = cost * norm(
                [space[m][l1,1], space[m][l1,2]]
                - [space[m][l2,1], space[m][l2,2]])
            end # l2 loop
        end # l1 loop
        push!(com_costs,comcost_l)
    end # m loop
    return com_costs
end

γ = γ_gen(para,space,nloc,1)

function rents_guess(para, nloc)
    @unpack M = para
    costs = []
    for m in 1:M
        push!(costs, ones(nloc[m], 1))
    end
    return costs
end

rents = rents_guess(para, nloc)

function prices_guess(JS)
    prices = ones(sum(JS))
    return prices
end

prices = prices_guess(JS)

function shock_gen(para, nloc, JS)
     @unpack σh, σw, σϵ, M, IU, IS = para
     ϵ = []
     εh = []
     εw = []
     ϵdist = Gumbel(0,σϵ)
     hdist = Gumbel(0,σh)
     wdist = Gumbel(0,σw)
     J = sum(JS)
     for m in 1:M
         push!(ϵ,rand(ϵdist,J,nloc[m]))
         push!(εh,rand(hdist,IU + IS,nloc[m]))
         push!(εw,rand(wdist,IS + IS,nloc[m]))
     end # m loop
     return ϵ, εh, εw
end

ϵ, εh, εw = shock_gen(para, nloc,JS)


function Pℓ(para, γ, firms, prices, nloc)
    @unpack τ, M, S, ζ = para
    P_l = []
    for m in 1:M
        Jm = firms[firms[:,2] .== m, :]
        Pml = ((τ.*γ[m][1:nloc[m],Jm[:,2]].+1).^(1+ζ) * (prices[Jm[:,1]]).^(1+ζ)).^(1/(1+ζ))
        push!(P_l, Pml)
    end # m loop
    return P_l
end

function wages_guess(para,nloc)
    @unpack S, M = para
    w = []
    for m in 1:M
        push!(w,10 .*ones(nloc[m],2,S))
    end # m loop
    return w
end

wages = wages_guess(para,nloc)


function household_sort(para, space, firms, prices, rents, wages, γ, n, εh, εw,nloc)
    @unpack αU, αS, ζ, η, τ, T, IU, IS, M, S, σw, σh = para
    E = [1, 2]
    I = [IU, IS]
    α = [αU, αS]
    # Construct Pl
    Pl = Pℓ(para, γ, firms, prices, nloc)
    # Blank array to hold max for each city
    maxm = zeros(IU + IS, 4, M)
    # Blank matrix to hold π^e_s(ℓ'|ℓ)
    πlpgl = []
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

    pop = zeros(M)
    for m in 1:M
        pop[m] = count(i -> i == m, Vmax[:,4])
    end # m loop
    pop = Int.(pop)

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

end

"""
MATRICES
firms: [city, location, sector]
prices: [price]
wages: w[m][ℓ,e,s]



"""

function fake_firm(para,nloc, JS)
    @unpack S, M = para
    J = sum(JS)
    city = Int.(rand(1:M, J))
    loc = Int.(zeros(J))
    for m in 1:M
        for j in 1:J
            if city[j] == m
                loc[j] = rand(1:nloc[m], 1)[1]
            end
        end # j loop
    end # m loop
    sec = Int.(rand(1:S, J))
    fake_firms = [city loc sec]
    return fake_firms
end

firms = fake_firm(para, nloc, JS)
