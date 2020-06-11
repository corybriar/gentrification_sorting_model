using Pkg
using LinearAlgebra
using Plots
using Distributions
using Parameters
using CSV
using DataFrames, DataFramesMeta, DelimitedFiles

cd("/Users/bigfoot13770/Documents/UO ECON PROGRAM/ADRIFT/gentrification_sorting_model/gentrification_sorting_model_R")



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

function rents_guess(para, nloc)
    @unpack M = para
    costs = []
    for m in 1:M
        push!(costs, ones(nloc[m], 1))
    end
    return costs
end

function prices_guess(para, nloc)
    @unpack M, S = para
    prices = []
    for m in 1:M
        push!(prices,ones(nloc[m],S))
    end # m loop
    return prices
end

function Pℓ(para, γ, firms, prices, ST, nloc)
    @unpack τ, M, S, ζ = para
    P_l = []
    for m in 1:M
        P_lm =[]
        for s in (ST+1):S
            Plms = (1 .+ τ.*γ[m]).^(1+ζ) * (firms[m][:,s] .* (prices[m][:,s]).^(1+ζ))
            push!(P_lm, Plms)
        end # s loop
        push!(P_l, sum(P_lm).^(1/(1+ζ)))
    end # m loop
    return P_l
end

function P_t(para, prices, firms, ST)
    @unpack M, β, αU, αS, ζ = para
    Pt = 0
    for s in 1:ST
        Ptm = 0
        for m in 1:M
            Ptm += sum(firms[m][:,s].*prices[m][:,s].^(1+ζ))
        end
        Pt = max(Ptm^(1\(1+ζ)),1e-16)
    end # s loop
    return Pt
end


function wages_guess(para,nloc)
    @unpack S, M = para
    w = []
    for m in 1:M
        push!(w, 10 .*ones(nloc[m],2,S))
    end # m loop
    return w
end

function household_sort(para, space, I, pop_old, firms, Pl, prices, rents, wages, γ, n, nloc)
    @unpack αU, αS, β, ζ, η, νU, νS, τ, T, M, S, σwU, σwS, σhU, σhS = para
    E = [1, 2]
    α = [αU, αS]
    σw = [σwU, σwS]
    σh = [σhU, σhS]
    ν = [νU, νS]

    # Blank matrix to hold π^e_s(ℓ'|ℓ)
    V = []
    sumV = []
    for m in 1:M
        Vesm = zeros(2, nloc[m],2,S)
            # Vesm[1,ℓ,e,s] = utility
            # Vesm[2,ℓ,e,s] = income
        sumVesm = zeros(2,S)

        for e in E
            # store endogenous amenities
            skillrat = zeros(nloc[m],1)
            if e == 1
                skillrat = sum(pop_old[m][:,1,:], dims = 3)[:,1,1]./sum(pop_old[m][:,2,:], dims = 3)[:,1,1]
            else
                skillrat = sum(pop_old[m][:,2,:], dims = 3)[:,1,1]./sum(pop_old[m][:,1,:], dims = 3)[:,1,1]
            end
            for s in 1:S
                VW = (1/σw[e]) .* (
                    log(n[e,s,m]) .+ log.(wages[m][:,e,s]')
                    .+ η.*log.(T .- n[e,s,m] .- γ[m])
                    )
                πm = exp.(VW) .* ((ones(1,nloc[m])*(exp.(VW))').^(-1))
                # fill Vesm with utilities for each location under sector s and education e
                Vesm[1,:,e,s] = transpose((1/σh[e]).*(diag(VW * transpose(πm)) .- α[e].*log.(rents[m]) .- (1-β)*(1-α[e]).*log.(Pl[m]))
                  .+ ν[e].*log.(skillrat[:]))
                # Back out incomes for those in ℓ ∈ Lm
                Vesm[2,:,e,s] = transpose(πm)*wages[m][:,e,s].*n[e,s,m]
                # sum up utilies across locations
                sumVesm[e,s] = sum(exp.(Vesm[1,:,e,s]))
            end # s loop
        end # e loop
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
            # household demands for housing, goods
            HXm[1,:,e,s] = α[e] .* popm[:,e,s] .* V[m][2,:,e,s] ./ rents[m][:]
            HXm[2,:,e,s] = (1-α[e]) .* popm[:,e,s] .* V[m][2,:,e,s] ./ Pl[m][:]
        end # e loop
        push!(pop, popm)
        push!(HX, HXm)
    end # m loop

    return V, pop, HX
end

function firm_sort(para, space, B, VH, JS, ST, Pl, Pt, pop, rents, wages, γ, ρ, θ, κ, n, nloc)
    @unpack αU, αS, β, ζ, η, τ, M, S, σϵ = para
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

    # calculate H^t before starting city loop
    Ht = zeros(ST)

    for m in 1:M, e in 1:2, s in 1:S
        Ht .+= sum(β.*(1-α[e]).* Pt^(-(1+ζ)).*(pop[m][:,e,s].*VH[m][2,:,e,s]))
    end # m-e loop
    for m in 1:M
        VFsm = zeros(nloc[m],S)
        Hlmes = zeros(nloc[m],2,S,S)
        factorsm = zeros(3,nloc[m],S)
        pricesm = zeros(nloc[m],S)
        # Compute demand potential for each ℓ,e,s triple
        for s in 1:S
            if s <=ST
                for e in 1:2, s2 in 1:S
                    Hlmes[:,e,s2,s] .= (1/(2*S))*Ht[s]
                end # e loop
            else
            for e in 1:2, s2 in 1:S
                Hlmes[:,e,s2,s] = (1-β).*(1 - α[e]).*(1 .+ τ.*γ[m]).^ζ * (pop[m][:,e,s2].*VH[m][2,:,e,s2] .* Pl[m].^(-(1 + ζ)))
            end # e loop
            end
        end # s loop
        # sum across columns of Hlmes to obtain H(ℓ)
        push!(Hl,sum(Hlmes, dims = (2,3))[:,1,1,:])
        for s in 1:S
            # assmeble unit cost function
            cl = (θ[s]^ρ[s].*wages[m][:,2,s].^(1-ρ[s]) + (1-θ[s])^ρ[s].*wages[m][:,1,s].^(1-ρ[s])).^(1/(1 - ρ[s]))
            VFsm[:,s] = (1/σϵ) .* (log.(B[m]) .- (1/(1 + ζ)).*log.(Hl[m][:,s]) - κ[s].*rents[m] - (1-κ[s]).*log.(cl))
            # optimal output for firm in m of type s across ℓ
            ysm = ((ζ/(1+ζ))./B[m] .*(rents[m]./κ[s]).^κ[s] .* (cl./(1-κ[s])).^(1-κ[s])).^ζ.*Hl[m][:,s]
            # Compute demand for U
            factorsm[1,:,s] = ysm./B[m] .*((1-θ[s])./wages[m][:,1,s]).^ρ[s].*(((1-κ[s])/κ[s]).*rents[m]).^κ[s].*cl.^(ρ[s]-κ[s])
            # Compute demand for S
            factorsm[2,:,s] = ysm./B[m] .*(θ[s]./wages[m][:,2,s]).^ρ[s].*(((1-κ[s])/κ[s]).*rents[m]).^κ[s].*cl.^(ρ[s]-κ[s])
            # Compute demand for K
            factorsm[3,:,s] = ysm./B[m] .*(((1-κ[s])/κ[s]).*rents[m]).^(κ[s]-1).*cl.^(1-κ[s])
            # store optimal prices
            pricesm[:,s] = ysm.^(1/ζ).*Hl[m][:,s].^(-1/ζ)
        end # s loop
        push!(VF,VFsm)
        push!(sumVF,sum(exp.(VFsm), dims = 1))
        push!(factors, factorsm)
        push!(prices,pricesm)
    end # m loop
    VFdenom = sum(sumVF)
    # Loop by city-sector to find sorting probabilities
    firms =[]
    factor = []
    for m in 1:M
        firmsm = zeros(nloc[m],S)
        factorsm = zeros(3,nloc[m],S)
        for s in 1:S
            firmsm[:,s] = JS[s] .* exp.(VF[m][:,s]) ./ (VFdenom[s])
            factorsm[1,:,s] = firmsm[:,s] .* factors[m][1,:,s]
            factorsm[2,:,s] = firmsm[:,s] .* factors[m][2,:,s]
            factorsm[3,:,s] = firmsm[:,s] .* factors[m][3,:,s]
        end
        push!(firms,firmsm)
        push!(factor,factorsm)
    end # m-s loop
    return firms, factor, prices, Hl
end


function form_rents(para, HX, factor, nloc, Rm, ψ)
    @unpack M, S = para
    rents = []
    for m in 1:M
        # Sum across household land demands by e and s, firm land by s
        rentm = Rm[m].*(sum(HX[m][1,:,:,:], dims = (2,3))[:,:,1] .+ sum(factor[m][3,:,:], dims = 2)).^ψ[m]
        push!(rents,rentm)
    end # m loop
    return rents
end


function form_wages(para, JS, wages_old, firms, factor, n, γ, nloc)
    @unpack T, ζ, η, M, S, σwU, σwS = para
    wages_new = []
    σw = [σwU, σwS]
    for m in 1:M
        wagesS = zeros(nloc[m],2,S)
        for s in 1:S, e in 1:2
            VW = (1/σw[e]) .* (
                log(n[e,s,m]) .+ log.(wages_old[m][:,e,s]')
                .+ η.*log.(T .- n[e,s,m] .- γ[m])
                )
            ugly_int = sum(exp.((1/σw[e]) .*(log(n[e,s,m]).+ η.*log.(T .- n[e,s,m] .- γ[m]))) .* ((ones(1,nloc[m])*(exp.(VW))').^(-1)), dims = 2)
            wagesS[:,e,s] = firms[m][:,s] .* (factor[m][e,:,s] .* n[e,s,m]^(-1) .* ugly_int.^(-1)).^σw[e]
        end # s-e loop
        push!(wages_new, wagesS)
    end # m loop
    return wages_new
end


function fake_firm(para,nloc, JS)
    @unpack S, M = para
    fake_firms = []
    for m in 1:M
        firmm = repeat([1/sum(nloc)].*JS,nloc[m])
        push!(fake_firms,firmm)
    end # m loop
    return fake_firms
end

function eq(para, space, I, γ, ρ, ψ, θ, κ, JS, ST, n, B, Rm, inner_max, outer_max, inner_tol, outer_tol, weights)
    @unpack M = para
    nloc = zeros(M)
    pop0 = []
    for m in 1:M
        nloc[m] = size(space[m])[1]
        push!(pop0, ones(Int(nloc[m]),2,4))
    end # m loop
    nloc = Int.(nloc)
    # initialize equilibrium objects
    wages = wages_guess(para, nloc)
    prices = prices_guess(para, nloc)
    rents = rents_guess(para, nloc)
    firms = fake_firm(para, nloc, JS)
    #γ = γ_gen(para,space,nloc,1)
    Pl = Pℓ(para, γ, firms, prices, ST, nloc)
    VH, pop, HX = household_sort(para, space, I, pop0, firms, Pl, prices, rents, wages, γ, n, nloc)

    # declare output vars
    VH_final = []
    pop_final = []
    HX_final = []
    firms_final = []
    factor_final = []
    prices_final = []
    wages_final = []
    rents_final = []
    VF_final =[]
    # initialize while loop
    iter = 0
    CONT = true
    wages_store = []
    rents_store = []
    diff_r = 0
    diff_w = 0
    # begin price equilibrium while loop
    while (iter < outer_max) & (CONT == true)
        iter += 1
        # initialize inner loop
        inner_it = 0
        inner_CONT = true
        hh_store = []
        firms_store = []
        # inner while loop to generate sorting equilibrium
        while (inner_it < inner_max) & (inner_CONT == true)
            inner_it += 1
            Pl = Pℓ(para, γ, firms, prices, ST, nloc)
            Pt = P_t(para, prices, firms, ST)
            # obtain household locations
            VH_new, pop_new, HX = household_sort(para, space, I, pop, firms, Pl, prices, rents, wages, γ, n, nloc)
            # given hh sort, obtain firm sort
            firms_new, factor, prices, VF = firm_sort(para,space, B, VH_new, JS, ST, Pl, Pt, pop_new, rents, wages, γ, ρ, θ, κ, n, nloc)
            # find max diff between cities
            diff_fer = zeros(M)
            diff_her = zeros(M)
            for m in 1:M
                firms_new[m] = max.(0.00000000000001, firms_new[m])
                pop[m] = max.(0.00000000000001, pop[m])
                diff_fer[m] = maximum(abs.(firms_new[m] - firms[m]))
                diff_her[m] = maximum(abs.(pop_new[m][:,:,:] - pop[m][:,:,:]))
            end # m loop
            diff_h = maximum(diff_her)
            diff_f = maximum(diff_fer)
            # iteration check
            if inner_it == 1
                # inner loop convergence check
                if (diff_f < inner_tol) & (diff_h < inner_tol)
                    firms_final = firms_new
                    factor_final = factor
                    prices_final = prices
                    VH_final = VH_new
                    pop_final = pop
                    HX_final = HX
                    VF_final = VF
                    inner_CONT = false
                    println("Sorting equilibrium obtained after $inner_it iterations")
                else
                   println("$inner_it, HH difference = $diff_h, firms difference = $diff_f")
                    firms = firms_new
                    pop = pop_new
                    push!(hh_store, pop)
                    push!(firms_store, firms)
                end
            elseif inner_it == 2
                # inner loop convergence check
                if (diff_f < inner_tol) & (diff_h < inner_tol)
                    firms_final = firms_new
                    factor_final = factor
                    prices_final = prices
                    VH_final = VH_new
                    pop_final = pop
                    HX_final = HX
                    VF_final = VF
                    inner_CONT = false
                    println("Sorting equilibrium obtained after $inner_it iterations")
                else
                    println("$inner_it, HH difference = $diff_h, firms difference = $diff_f")
                    firms = 0.5 .* (firms_new + firms)
                    pop = 0.5 .* (pop_new + pop)
                    push!(hh_store, pop)
                    push!(firms_store, firms)
                end
            elseif inner_it == 3
                # inner loop convergence check
                if (diff_f < inner_tol) & (diff_h < inner_tol)
                    firms_final = firms_new
                    factor_final = factor
                    prices_final = prices
                    VH_final = VH_new
                    pop_final = pop
                    HX_final = HX
                    VF_final = VF
                    inner_CONT = false
                    println("Sorting equilibrium obtained after $inner_it iterations")
                else
                    println("$inner_it, HH difference = $diff_h, firms difference = $diff_f")
                    firms = (1/3) .* (firms_new + firms + firms_store[inner_it - 2])
                    pop = (1/3) .* (pop_new + pop + hh_store[inner_it - 2])
                    push!(hh_store, pop)
                    push!(firms_store, firms)
                end
            else
                # inner loop convergence check
                if (diff_f < inner_tol) & (diff_h < inner_tol)
                    firms_final = firms_new
                    factor_final = factor
                    prices_final = prices
                    VH_final = VH_new
                    pop_final = pop
                    HX_final = HX
                    VF_final = VF
                    inner_CONT = false
                    println("Sorting equilibrium obtained after $inner_it iterations")
                else
                    println("$inner_it, HH difference = $diff_h, firms difference = $diff_f")
                    firms = weights[1].*firms_new + weights[2].*firms + weights[3].*firms_store[inner_it - 2] + weights[4].* firms_store[inner_it - 3]
                    pop = weights[1].*pop_new + weights[2].*pop + weights[3].*hh_store[inner_it - 2] + weights[4].*hh_store[inner_it - 3]
                    push!(hh_store, pop)
                    push!(firms_store, firms)
                end
            end # iteration check
        end # sorting loop
        # form wages and rents based on sorting
        wages_new = form_wages(para, JS, wages, firms_final, factor_final, n, γ, nloc)
        rents_new = form_rents(para, HX, factor_final, nloc, Rm, ψ)
        # find largest difference between cities
        diff_wers = zeros(M)
        diff_rers = zeros(M)
        for m in 1:M
            diff_wers[m] = maximum(abs.(wages_new[m] - wages[m]))
            diff_rers[m] = maximum(abs.(rents_new[m] - rents[m]))
        end
        diff_r = maximum(diff_rers)
        diff_w = maximum(diff_wers)
        # check for convergence
        if iter == 1
            if (diff_w < outer_tol) & (diff_r < outer_tol)
                wages_final = wages_new
                rents_final = rents_new
                CONT = false
                println("Wage-rent equilibrium obtained after $iter iterations")
            else
                println("$iter Wage difference = $diff_w, Rent difference = $diff_r")
                wages = wages_new
                rents = rents_new
                push!(wages_store, wages)
                push!(rents_store, rents)
            end
        elseif iter == 2
            if (diff_w < outer_tol) & (diff_r < outer_tol)
                wages_final = wages_new
                rents_final = rents_new
                CONT = false
                println("Wage-rent equilibrium obtained after $iter iterations")
            else
                println("$iter Wage difference = $diff_w, Rent difference = $diff_r")
                wages = 0.5.*(wages_new + wages)
                rents = 0.5.*(rents_new + rents)
                push!(wages_store, wages)
                push!(rents_store, rents)
            end
        elseif iter == 3
            if (diff_w < outer_tol) & (diff_r < outer_tol)
                wages_final = wages_new
                rents_final = rents_new
                CONT = false
                println("Wage-rent equilibrium obtained after $iter iterations")
            else
                println("$iter Wage difference = $diff_w, Rent difference = $diff_r")
                wages = (wages_new + wages + wages_store[1])./3
                rents = (rents_new + rents + rents_store[1])./3
                push!(wages_store, wages)
                push!(rents_store, rents)
            end
        else
            if (diff_w < outer_tol) & (diff_r < outer_tol)
                wages_final = wages_new
                rents_final = rents_new
                CONT = false
                println("Wage-rent equilibrium obtained after $iter iterations")
            else
                println("$iter Wage difference = $diff_w, Rent difference = $diff_r")
                wages = weights[1].*wages_new + weights[2].*wages + weights[3].*wages_store[iter - 2] + weights[4].*wages_store[iter - 3]
                rents = weights[1].*rents_new + weights[2].*rents + weights[3].*rents_store[iter - 2] + weights[4].*rents_store[iter - 3]
                push!(wages_store, wages)
                push!(rents_store, rents)
            end
        end # iteration check
        wages_final = wages_new
        rents_final = rents_new
    end # price loop
    return VH_final, pop_final, HX_final, firms_final, factor_final, prices_final, wages_final, rents_final, VF_final
end

ρ = [2 2 2 2]/1.4
ψ = [1.136 0.806]

#(U,t) (S,t) | (U,nt) (S,nt)
θ90 = [0.233 0.398 0.278 0.538]
θ18 = [0.304 0.489 0.383 0.579]
κ90 = 1 .- [0.579 0.550 0.632 0.813]
κ18 = 1 .- [0.491 0.557 0.635 0.805]
JS94 = [18321 84865 23316 20083]/1000
JS17 = [26384 81267 54414 28839]/1000

I90 = [3471 951]
I18 = [3614 2165]

# n[e,s,m]
n90 = 8 .*ones(2,4,2)
n18 = 8 .*ones(2,4,2)
n18[2,:,:] .= 10
#n[1,:,:] .= 8
Rm = [0.01 0.01]
ST = 2

nloc = [186 241]
B = []
for m in 1:M
    push!(B, 10 .*ones(nloc[m]))
end
# B[1][1] = 10

n90 = 7.66.*ones(2,4,2)
n90[2,1,:] .= 8.46
n90[1,2,:] .= 7.11
n90[2,2,:] .= 8.44
n90[1,3,:] .= 6.88
n90[2,3,:] .= 8.11
n90[1,4,:] .= 6.33
n90[2,4,:] .= 7.78

n18 = 8.39.*ones(2,4,2)
n18[2,1,:] .= 8.57
n18[1,2,:] .= 8.02
n18[2,2,:] .= 8.60
n18[1,3,:] .= 7.49
n18[2,3,:] .= 8.17
n18[1,4,:] .= 7.07
n18[2,4,:] .= 8.19

weights = [0.4 0.3 0.2 0.1]
inner_max = 400
outer_max = 400
inner_tol = 0.01
outer_tol = 0.01

space =[]
sea_cent = CSV.read("sea_cent.csv")
sea_cent = convert(Matrix, sea_cent)
push!(space, sea_cent)
det_cent = CSV.read("det_cent.csv")
det_cent = convert(Matrix, det_cent)
push!(space, det_cent)

γ = []
γ_sea = CSV.read("sea_time.csv")
γ_sea = convert(Matrix, γ_sea)
γ_sea = reshape(γ_sea, (186, 186))
push!(γ, 2 .*γ_sea)
γ_det = CSV.read("det_time.csv")
γ_det = convert(Matrix, γ_det)
γ_det = reshape(γ_det, (241, 241))
push!(γ, 2 .*γ_det)

@with_kw struct parameters
    αU::Float64 = 0.337   # Utility parameter
    αS::Float64 = 0.321
    β::Float64 = 0.4
    ζ::Float64 = -2.5    # Elasticity of substitution between goods
    η::Float64 = 1.5    # Elast. of utility to work/commute
    νU::Float64 = 0
    νS::Float64 = 0
    τ::Float64 = 0.1     # Travel cost parameter
    T::Float64 = 20
    M::Int64 = 2        # Number of cities
    S::Int64 = 4       # Number of sectors
    σhU::Float64 = 0.475     # sd of εh
    σhS::Float64 = 0.112     # sd of εh
    σwU::Float64 = 0.3
    σwS::Float64 = 0.5
    σϵ::Float64 = 0.4  # sd of ϵ
end

para = parameters()

# Simulation for 1990
VH_eq90, pop_eq90, HX_eq90, firms_eq90, factor_eq90, prices_eq90, wages_eq90, rents_eq90, VF90 = eq(para, space, I90, γ, ρ, ψ, θ90, κ90, JS94, ST, n90, B, Rm, inner_max, outer_max, inner_tol, outer_tol, weights)
# Simulation for 2018
VH_eq18, pop_eq18, HX_eq18, firms_eq18, factor_eq18, prices_eq18, wages_eq18, rents_eq18, VF18 = eq(para, space, I18, γ, ρ, ψ, θ18, κ18, JS17, ST, n18, B, Rm, inner_max, outer_max, inner_tol, outer_tol, weights)

# write files for 90's sim
CSV.write("sea_sim_pop90.csv", convert(DataFrame, sum(pop_eq90[1], dims = 3)[:,:,1]))
CSV.write("det_sim_pop90.csv", convert(DataFrame, sum(pop_eq90[2], dims = 3)[:,:,1]))
CSV.write("sea_sim_firms90.csv", convert(DataFrame, firms_eq90[1]))
CSV.write("det_sim_firms90.csv", convert(DataFrame, firms_eq90[2]))

CSV.write("sea_sim_pop18.csv", convert(DataFrame, sum(pop_eq18[1], dims = 3)[:,:,1]))
CSV.write("det_sim_pop18.csv", convert(DataFrame, sum(pop_eq18[2], dims = 3)[:,:,1]))
CSV.write("sea_sim_firms18.csv", convert(DataFrame, firms_eq18[1]))
CSV.write("det_sim_firms18.csv", convert(DataFrame, firms_eq18[2]))


@unpack M = para

space = []
push!(space, [0 0; 0 1; 0 2])
nloc = [3]
γ = γ_gen(para, space, nloc, 1)

n = 10 .* ones(2,4,1)
wages_old = wages
JS = [10 10 10 10]
I = [200 100]

nloc = zeros(M)
pop0 = []
for m in 1:M
    nloc[m] = size(space[m])[1]
    push!(pop0, ones(Int(nloc[m]),2,4))
end # m loop
nloc = Int.(nloc)
θ = [0.3 0.7 0.3 0.7]
κ = [0.3 0.3 0.3 0.3]
B = []
push!(B,ones(3))
# initialize equilibrium objects
wages_old = wages_guess(para, nloc)
prices = prices_guess(para, nloc)
rents = rents_guess(para, nloc)
firms = fake_firm(para, nloc, JS)
#γ = γ_gen(para,space,nloc,1)
Pt = P_t(para, prices, firms, ST)
Pl = Pℓ(para, γ, firms, prices, ST, nloc)
VH, pop, HX = household_sort(para, space, I, pop0, firms, Pl, prices, rents, wages, γ, n, nloc)
VH_new, pop_new, HX = household_sort(para, space, I, pop, firms, Pl, prices, rents, wages, γ, n, nloc)
# given hh sort, obtain firm sort
firms_new, factor, prices, VF = firm_sort(para,space, B, VH_new, JS, ST, Pl, Pt, pop_new, rents, wages, γ, ρ, θ, κ, n, nloc)


VW = (1/σw[e]) .* (
    log(n[e,s,m]) .+ log.(wages_old[m][:,e,s]')
    .+ η.*log.(T .- n[e,s,m] .- γ[m])
    )
ugly_int = sum(exp.((1/σw[e]) .*(log(n[e,s,m]).+ η.*log.(T .- n[e,s,m] .- γ[m]))) .* ((ones(1,nloc[m])*(exp.(VW))').^(-1)), dims = 2)

firms_new[m][:,s] .* (factor[m][e,:,s] .* n[e,s,m]^(-1) .* ugly_int.^(-1)).^σw[e]
