
firm_rat = JS17./JS94

function firm_sort_alt(para, space, B, VH, JS, ST, Pl, Pt, pop, rents, wages, γ, ρ, θ, κ, n, nloc, firms90, firm_rat)
    @unpack αU, αS, β, ζ, η, τ1, τ2, M, S, σϵ = para
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
                Hlmes[:,e,s2,s] = (1-β).*(1 - α[e]).*(1 .+ τ1.*γ[m].^τ2).^ζ * (pop[m][:,e,s2].*VH[m][2,:,e,s2] .* Pl[m].^(-(1 + ζ)))
            end # e loop
            end
        end # s loop
        # sum across columns of Hlmes to obtain H(ℓ)
        push!(Hl,sum(Hlmes, dims = (2,3))[:,1,1,:])
        for s in 1:S
            # assmeble unit cost function
            cl = (θ[s]^ρ[s].*wages[m][:,2,s].^(1-ρ[s]) + (1-θ[s])^ρ[s].*wages[m][:,1,s].^(1-ρ[s])).^(1/(1 - ρ[s]))
            VFsm[:,s] = (1/σϵ) .* (log.(B[m]) .- (1/(1 + ζ)).*log.(Hl[m][:,s]) - κ[s].*log.(rents[m]) - (1-κ[s]).*log.(cl))
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
            firmsm[:,s] = firms90[m][:,s] .* firm_rat[s]
            factorsm[1,:,s] = firmsm[:,s] .* factors[m][1,:,s]
            factorsm[2,:,s] = firmsm[:,s] .* factors[m][2,:,s]
            factorsm[3,:,s] = firmsm[:,s] .* factors[m][3,:,s]
        end
        push!(firms,firmsm)
        push!(factor,factorsm)
    end # m-s loop
    return firms, factor, prices, Hl, VF
end


function eq_alt(para, space, N, γ, ρ, ψ, θ, κ, JS, WP, ST, n, B, Rm, inner_max, outer_max, inner_tol, outer_tol, weights, firms90)
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
    firms = fake_firm(para,nloc,JS)
    #firms = []
    #push!(firms, [firms90[1][:,1].*firm_rat[1] firms90[1][:,2].*firm_rat[2] firms90[1][:,3].*firm_rat[3] firms90[1][:,4].*firm_rat[4]] )
    #push!(firms, [firms90[2][:,1].*firm_rat[1] firms90[2][:,2].*firm_rat[2] firms90[2][:,3].*firm_rat[3] firms90[2][:,4].*firm_rat[4]])
    #γ = γ_gen(para,space,nloc,1)
    Pl = Pℓ(para, γ, firms, prices, ST, nloc)
    VH, pop, HX = household_sort(para, space, N, pop0, firms, Pl, prices, rents, wages, γ, n, nloc)

    # declare output vars
    VH_final = []
    pop_final = []
    HX_final = []
    firms_final = []
    factor_final = []
    prices_final = []
    wages_final = []
    rents_final = []
    Hl_final =[]
    VF_final = []
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
            VH_new, pop_new, HX = household_sort(para, space, N, pop, firms, Pl, prices, rents, wages, γ, n, nloc)
            # given hh sort, obtain firm sort
            firms_new, factor, prices, Hl, VF = firm_sort_alt(para, space, B, VH, JS, ST, Pl, Pt, pop, rents, wages, γ, ρ, θ, κ, n, nloc, firms90, firm_rat)
            #firms_new, factor, prices, Hl, VF = firm_sort(para,space, B, VH_new, JS, ST, Pl, Pt, pop_new, rents, wages, γ, ρ, θ, κ, n, nloc)
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
                    Hl_final = Hl
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
                    Hl_final = Hl
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
                    Hl_final = Hl
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
                    Hl_final = Hl
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
        wages_new = form_wages(para, JS, WP, wages, firms_final, factor_final, n, γ, nloc)
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
    return VH_final, pop_final, HX_final, firms_final, factor_final, prices_final, wages_final, rents_final, Hl_final, VF_final
end


# Simulation for 2018
VH_eq18, pop_eq18, HX_eq18, firms_eq18, factor_eq18, prices_eq18, wages_eq18, rents_eq18, Hl18, VF18 = eq_alt(para, space, I90, γ, ρ, ψ, θ90, κ90, JS94, WP90, ST, n90, B, Rm, inner_max, outer_max, inner_tol, outer_tol, weights, firms_eq90)


CSV.write("sea_sim_pop18freeze.csv", convert(DataFrame, sum(1000 .*pop_eq18[1], dims = 3)[:,:,1]))
CSV.write("det_sim_pop18freeze.csv", convert(DataFrame, sum(1000 .*pop_eq18[2], dims = 3)[:,:,1]))
CSV.write("sea_sim_firms18freeze.csv", convert(DataFrame, 1000 .*firms_eq18[1]))
CSV.write("det_sim_firms18freeze.csv", convert(DataFrame, 1000 .*firms_eq18[2]))

CSV.write("sea_sim_rents18.csv", convert(DataFrame, rents_eq18[1]))
CSV.write("det_sim_rents18.csv", convert(DataFrame, rents_eq18[2]))


Pl = Pℓ(para, γ, firms_eq90, prices_eq90, ST, nloc)

VH_new, pop_new, HX = household_sort(para, space, I90, pop_eq90, firms_eq90, Pl, prices_eq90, rents_eq90, wages_eq90, γ, n, nloc)
