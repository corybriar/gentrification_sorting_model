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
κ = [0.3, 0.3, 0.3, 0.3]
JS = [15,15,15,15]
n = 10 .*ones(2,4,2)
B = [1, 1]

nloc = [15 15]
# create imaginary space to work with
"""function space_gen(para,nloc, sizemax)
    @unpack M = para
    space = []
    space_dist = Uniform(0,sizemax)
    for m in 1:M
        locs = rand(space_dist, nloc[m], 2)
        push!(space, locs)
    end # m loop
    return space
end

space = space_gen(para,nloc,1)"""

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
                - [space[m][l2,1], space[m][l2,2]]) .+ 1
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

rents_0 = rents_guess(para, nloc)

function prices_guess(para, nloc)
    @unpack M, S = para
    prices = []
    for m in 1:M
        push!(prices,ones(nloc[m],S))
    end # m loop
    return prices
end

prices_0 = prices_guess(para, nloc)

"""function shock_gen(para, nloc, JS)
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

ϵ, εh, εw = shock_gen(para, nloc,JS)"""


function Pℓ(para, γ, firms, prices, nloc)
    @unpack τ, M, S, ζ = para
    P_l = []
    for m in 1:M
        P_lm =[]
        for s in 1:S
            Plms = (γ[m]).^ζ * (firms[m][:,s] .* (τ.*prices[m][:,s]).^(1+ζ))
            push!(P_lm, Plms)
        end # s loop
        push!(P_l, sum(P_lm))
    end # m loop
    return P_l
end
Pl = Pℓ(para, γ, firms, prices, nloc)

function wages_guess(para,nloc)
    @unpack S, M = para
    w = []
    for m in 1:M
        push!(w,10 .*ones(nloc[m],2,S))
    end # m loop
    return w
end

wage_0 = wages_guess(para,nloc)



function household_sort(para, space, firms, prices, rents, wages, γ, n, nloc)
    @unpack αU, αS, ζ, η, τ, T, IU, IS, M, S, σw, σh = para
    E = [1, 2]
    I = [IU, IS]
    α = [αU, αS]
    # Construct P(ℓ)
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
            Vesm[2,:,e,s] = transpose(πm)*wages[m][:,e,s].*n[e,s,m]
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
        # HX[m][1,ℓ,e,s] = H
        # HX[m][2,ℓ,e,s] = X
        for e in 1:2, s in 1:S
            popm[:,e,s] = I[e] .* exp.(V[m][1,:,e,s]) .* (sum(Vdenom[e,:])).^(-1)
            #
            HXm[1,:,e,s] = α[e] .* popm[:,e,s] .* V[m][2,:,e,s] ./ rents[m][:]
            HXm[2,:,e,s] = (1-α[e]) .* popm[:,e,s] .* V[m][2,:,e,s] ./ Pl[m][:]
        end # e loop
        push!(pop, popm)
        push!(HX, HXm)
    end # m loop

    return V, pop, HX
end

VH, pop, HX = household_sort(para, space, firms, prices, rents, wages, γ, n, nloc)


function firm_sort(para,space, VH, JS, Pl, rents, wages, ρ, θ, n, nloc)
    @unpack αU, αS, ζ, η, τ, M, S, σϵ = para
    α = [αU, αS]
    VF =[]
    Hl = []
    sumVF = []
    factors =[]
    # factors[m][1,ℓ,s] = U
    # factors[m][2,ℓ,s] = S
    # factors[m][3,ℓ,s] = K
    prices = []
    # prices[m] = [ℓ,s]
    for m in 1:M
        VFsm = zeros(nloc[m],S)
        Hlmes = zeros(nloc[m],2,S)
        factorsm = zeros(3,nloc[m],S)
        pricesm = zeros(nloc[m],S)
        # Compute demand potential for each ℓ,e,s triple
        for s in 1:S
            for e in 1:2
                Hlmes[:,e,s] = (1 - α[e]).*(τ.*γ[m])^ζ * (pop[m][:,e,s].*VH[m][2,:,e,s]  .* Pl[m].^(-(1 + ζ)))
            end # e loop
        end # s loop
        # sum across columns of Hlmes to obtain H(ℓ)
        push!(Hl,sum(Hlmes, dims = (2,3))[:,1,1])
        for s in 1:S
            # assmeble unit cost function
            cl = (θ[s]^ρ[s].*wages[m][:,2,s].^(1-ρ[s]) + (1-θ[s])^ρ[s].*wages[m][:,1,s].^(1-ρ[s])).^(1/(1 - ρ[s]))
            VFsm[:,s] = (1/σϵ) .* (-(1/(1 + ζ)).*log.(Hl[m]) - κ[s].*rents[m] - (1-κ[s]).*log.(cl))
            # optimal output for firm in m of type s across ℓ
            ysm = ((ζ/((1+ζ)B[m])).*(rents[m]./κ[s]).^κ[s] .* (cl./(1-κ[s])).^(1-κ[s])).^ζ.*Hl[m]
            # Compute demand for U
            factorsm[1,:,s] = ysm./B[m].*((1-θ[s])./wages[m][:,1,s]).^ρ[s].*(((1-κ[s])/κ[s]).*rents[m]).^κ[s].*cl.^(ρ[s]-κ[s])
            # Compute demand for S
            factorsm[2,:,s] = ysm./B[m].*(θ[s]./wages[m][:,2,s]).^ρ[s].*(((1-κ[s])/κ[s]).*rents[m]).^κ[s].*cl.^(ρ[s]-κ[s])
            # Compute demand for K
            factorsm[3,:,s] = ysm./B[m].*(((1-κ[s])/κ[s]).*rents[m]).^(-κ[s]).*cl.^(1-κ[s])
            # store optimal prices
            pricesm[:,s] = ysm.^(1/ζ).*Hl[m].^(-1/ζ)
        end # s loop
        push!(VF,VFsm)
        push!(sumVF,sum(exp.(VFsm), dims = 1))
        push!(factors, factorsm)
        push!(prices,pricesm)
    end # m loop
    VFdenom = sum(sumVF)
    # Loop by city-sector to find sorting probabilities
    firms =[]
    for m in 1:M
        firmsm = zeros(nloc[m],S)
        for s in 1:S
            firmsm[:,s] = JS[s] .* exp.(VF[m][:,s]) ./ (VFdenom[s])
        end
        push!(firms,firmsm)
    end # m-s loop

    return firms, factors, prices
end

firms, factor, prices = firm_sort(para,space, VH, JS, Pl, rents, wages, ρ, θ, n, nloc)

function form_rents(para, HX, factor, nloc, ψ)
    @unpack M, S = para
    rents = []
    for m in 1:M
        # Sum across household land demands by e and s, firm land by s
        rentm = (sum(HX[m][1,:,:,:], dims = (2,3))[:,:,1] .+ sum(factor[m][3,:,:], dims = 2)).^ψ[m]
        push!(rents,rentm)
    end # m loop
    return rents
end

rents_new = form_rents(para, HX, factor, nloc, ψ)

function form_wages(para, JS, wages_old, firms, factor, n, γ, nloc)
    @unpack T, ζ, η, M, S, σw = para
    wages_new = []
    for m in 1:M
        wagesS = zeros(nloc[m],2,S)
        for s in 1:S, e in 1:2
            VW = (1/σw) .* (
                log(n[e,s,m]) .+ log.(wages_old[m][:,e,s]')
                .+ η.*log.(T .- n[e,s,m] .- γ[m])
                )
            ugly_int = sum(exp.((1/σw) .*(log(n[e,s,m]).+ η.*log.(T .- n[e,s,m] .- γ[m]))) .* ((ones(1,nloc[m])*(exp.(VW))').^(-1)), dims = 2)
            wagesS[:,e,s] = firms[m][:,s] .* (factor[m][e,:,s] .* n[e,s,m] .* ugly_int.^(-1)).^σw
        end # s-e loop
        push!(wages_new, wagesS)
    end # m loop
    return wages_new
end

wages_new = form_wages(para, JS, wages, firms, factor, n, γ, nloc)

"""
MATRICES
firms: [city, location, sector]
prices: [price]
wages: w[m][ℓ,e,s]



"""

function fake_firm(para,nloc, JS)
    @unpack S, M = para
    fake_firms = []
    for m in 1:M
        firmm = repeat(transpose([1/sum(nloc)].*JS),nloc[m])
        push!(fake_firms,firmm)
    end # m loop
    return fake_firms
end

firms_0 = fake_firm(para, nloc, JS)

ρ = [2 3 4 5]
ψ = [1.5, 1.5]
θ = [0.3, 0.4, 0.5, 0.6]
κ = [0.3, 0.3, 0.3, 0.3]
JS = [15,15,15,15]
n = 10 .*ones(2,4,2)
B = [1, 1]

nloc = [15 15]

weights = [0.4, 0.3, 0.2, 0.1]
inner_max = 200
outer_max = 200
inner_tol = 0.1
outer_tol = 0.1

function eq(para, ρ, ψ, θ, κ, JS, n, B, nloc, inner_max, outer_max, inner_tol, outer_tol, weights)
    # initialize distribution of firms, wages, rents, prices
    wages = wages_guess(para, nloc)
    prices = prices_guess(para, nloc)
    rents = rents_guess(para, nloc)
    firms = fake_firm(para, nloc, JS)

    # initialize while loop
    iter = 0
    CONT = true
    wages_store = []
    #prices_store = []
    rents_store = []

    # begin price equilibrium while loop
    while (iter < outer_max) & (CONT == true)
        iter += 1

        # initialize inner loop
        inner_it = 0
        inner_CONT = true
        hh_store = []
        firms_store = []
        # inner while loop to generate sorting equilibrium
        while (inner_it < inner_max) &
            inner_it += 1
            Pl = Pℓ(para, γ, firms, prices, nloc)
            # obtain household locations
            VH_new, pop, HX = household_sort(para, space, firms, prices, rents, wages, γ, n, nloc)
            # given hh sort, obtain firm sort
            firms_new, factor, prices = firm_sort(para,space, VH, JS, Pl, rents, wages, ρ, θ, n, nloc)
            # iteration check
            if inner_it == 1
                # inner loop convergence check
                if (max(firms_new - firms) < inner_tol) & (max(VH_new[:][1,:,:,:] - VH[:][1,:,:,:]) < inner_tol)
                    firms = firms_new
                    VH = VH_new
                    inner_CONT = false
                    println("Sorting equilibrium obtained after $inner_it iterations")
                else
                    println("$inner_it, HH difference = "max(firms_new - firms)" , firms difference = "max(VH_new[:][1,:,:,:] - VH[:][1,:,:,:]))
                    firms = firms_new
                    VH = VH_new
                    push!(hh_store, VH)
                    push!(firms_store, firms)
                end
            elseif inner_it == 2
                # inner loop convergence check
                if (max(firms_new - firms) < inner_tol) & (max(VH_new[:][1,:,:,:] - VH[:][1,:,:,:]) < inner_tol)
                    firms = firms_new
                    VH = VH_new
                    inner_CONT = false
                    println("Sorting equilibrium obtained after $inner_it iterations")
                else
                    println("$inner_it, HH difference = "max(firms_new - firms)" , firms difference = "max(VH_new[:][1,:,:,:] - VH[:][1,:,:,:]))
                    firms = 0.5 .* (firms_new + firms)
                    VH = 0.5 .* (VH_new + VH)
                    push!(hh_store, VH)
                    push!(firms_store, firms)
                end
            elseif inner_it == 3
                # inner loop convergence check
                if (max(firms_new - firms) < inner_tol) & (max(VH_new[:][1,:,:,:] - VH[:][1,:,:,:]) < inner_tol)
                    firms = firms_new
                    VH = VH_new
                    inner_CONT = false
                    println("Sorting equilibrium obtained after $inner_it iterations")
                else
                    println("$inner_it, HH difference = "max(firms_new - firms)" , firms difference = "max(VH_new[:][1,:,:,:] - VH[:][1,:,:,:]))
                    firms = (1/3) .* (firms_new + firms + firms_store[inner_it - 2])
                    VH = (1/3) .* (VH_new + VH + hh_store[inner_it - 2])
                    push!(hh_store, VH)
                    push!(firms_store, firms)
                end
            else
                # inner loop convergence check
                if (max(firms_new - firms) < inner_tol) & (max(VH_new[:][1,:,:,:] - VH[:][1,:,:,:]) < inner_tol)
                    firms = firms_new
                    VH = VH_new
                    inner_CONT = false
                    println("Sorting equilibrium obtained after $inner_it iterations")
                else
                    println("$inner_it, HH difference = "max(firms_new - firms)" , firms difference = "max(VH_new[:][1,:,:,:] - VH[:][1,:,:,:]))
                    firms = weights[1].*firms_new + weights[2].*firms + weights[3].*firms_store[inner_it - 2] + weights[4].* firms_store[inner_it - 3]
                    VH = weights[1].*VH_new + weights[2].*VH + weights[3].*hh_store[inner_it - 2] + weights[4].*hh_store[inner_it - 3]
                    push!(hh_store, VH)
                    push!(firms_store, firms)
                end
            end # iteration check
        end # sorting loop
        # form wages and rents based on sorting
        wages_new = form_wages(para, JS, wages, firms, factor, n, γ, nloc)
        rents_new = form_rents(para, HX, factor, nloc, ψ)
        # check for convergence
        if iter == 1
            if (max(wages_new - wages) < outer_tol) & (max(rents_new - rents) < outer tol)
                wages = wages_new
                rents = rents_new
                CONT = false
                println("Wage-rent equilibrium obtained after $iter iterations")
            else
                println("$iter Wage difference = "max(wages_new - wages)", Rent difference = "max(rents_new - rents))
                wages = wages_new
                rents = rents_new
                push!(wages_store,wages)
                push!(rents_store, rents)
            end
        elseif iter == 2
            if (max(wages_new - wages) < outer_tol) & (max(rents_new - rents) < outer tol)
                wages = wages_new
                rents = rents_new
                CONT = false
                println("Wage-rent equilibrium obtained after $iter iterations")
            else
                println("$iter Wage difference = "max(wages_new - wages)", Rent difference = "max(rents_new - rents))
                wages = 0.5.*(wages_new + wages)
                rents = 0.5.*(rents_new + rents)
                push!(wages_store,wages)
                push!(rents_store, rents)
            end
        elseif iter == 3
            if (max(wages_new - wages) < outer_tol) & (max(rents_new - rents) < outer tol)
                wages = wages_new
                rents = rents_new
                CONT = false
                println("Wage-rent equilibrium obtained after $iter iterations")
            else
                println("$iter Wage difference = "max(wages_new - wages)", Rent difference = "max(rents_new - rents))
                wages = (wages_new + wages + wages_store[1])./3
                rents = (rents_new + rents + rents_store[1])./3
                push!(wages_store,wages)
                push!(rents_store, rents)
            end
        else
            if (max(wages_new - wages) < outer_tol) & (max(rents_new - rents) < outer tol)
                wages = wages_new
                rents = rents_new
                CONT = false
                println("Wage-rent equilibrium obtained after $iter iterations")
            else
                println("$iter Wage difference = "max(wages_new - wages)", Rent difference = "max(rents_new - rents))
                wages = weights[1].*wages_new + weights[2].*wages + weights[3].*wages_store[iter - 2] + weights[4].*wages_store[iter - 3]
                rents = weights[1].*rents_new + weights[2].*rents + weights[3].*rents_store[iter - 2] + weights.*rents_store[iter - 3]
                push!(wages_store,wages)
                push!(rents_store, rents)
            end
        end # iteration check
    end # price loop
    return VH, pop, HX, firms_new, factor, prices, wages, rents
end
