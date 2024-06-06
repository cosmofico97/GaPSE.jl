# -*- encoding: utf-8 -*-
#
# This file is part of GaPSE
# Copyright (C) 2022 Matteo Foglieni
#
# GaPSE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GaPSE is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GaPSE. If not, see <http://www.gnu.org/licenses/>.
#



"""
    derivate_point(xp, yp, x1, y1, x2, y2)

Return the derivative in `(xp, yp)`, given the neighboor points
`(x1,y1)` and `(x2,y2)`, with `x1 < xp < x2`.
It is not assumed that `x2 - xp = xp - x1`.


See also: [`derivate_vector`](@ref), [`spectral_index`](@ref)]
"""
function derivate_point(xp, yp, x1, y1, x2, y2)
    l2, l1 = (x2 - xp), (xp - x1)
    m2, m1 = (y2 - yp) / l2, (yp - y1) / l1
    res = (m1 * l2 + m2 * l1) / (l1 + l2)
    #println(res)
    return res
end


"""
    derivate_vector(XS, YS; N::Int=1)

Apply `derivate_point` to the vector.
`N::Int=1` is the number of points at each edge that idk.

See also: [`derivate_point`](@ref), [`spectral_index`](@ref)]
"""
function derivate_vector(XS, YS; N::Int=1)
    @assert length(XS) == length(YS) "xs and ys must have the same length!"
    @assert length(YS) > 2 * N "length of xs and ys must be > 2N !"

    mean_exp_xs = sum([log10(abs(x)) for x in XS]) / length(XS)
    en_xs = 10.0^(-mean_exp_xs)
    xs = XS .* en_xs

    mean_exp_ys = sum([log10(abs(y)) for y in YS]) / length(YS)
    en_ys = 10.0^(-mean_exp_ys)
    ys = YS .* en_ys

    mean_ys = sum(ys) / length(ys)
    @assert !all([isapprox(y / mean_ys, 1.0, rtol=1e-6) for y in ys]) "DO NOT WORK!"

    if N == 1
        real_vec = [derivate_point(xs[i], ys[i], xs[i-1], ys[i-1], xs[i+1], ys[i+1])
                    for i in (N+1):(length(xs)-N)]
        return vcat(real_vec[begin], real_vec, real_vec[end]) .* (en_xs / en_ys)
    else
        vec = [derivate_point(xs[i], ys[i], xs[i-j], ys[i-j], xs[i+j], ys[i+j])
                for i in (N+1):(length(xs)-N), j in 1:N]
        real_vec = [sum(row) / N for row in eachrow(vec)]
        return vcat([real_vec[begin] for i in 1:N], real_vec,
            [real_vec[end] for i in 1:N]) .* (en_xs / en_ys)
    end
end


"""
    spectral_index(xs, ys; N::Int=1, con=false)

Return the spectral index vector for the two input vectors.

The spectral index ``S`` of a generic function ``f = f(x)`` is
defined as:
```math
    S = \\frac{\\partial \\log f(x)}{\\partial \\log x} 
        = \\frac{x}{f(x)} \\frac{\\partial f(x)}{\\partial x} 
```

See also: [`derivate_point`](@ref), [`derivate_vector`](@ref)
"""
function spectral_index(xs, ys; N::Int=1, con=false)
    derivs = derivate_vector(xs, ys; N=N)

    if con == false
        res = [x * d / y for (x, y, d) in zip(xs, ys, derivs)]
        return vcat(
            [res[begin+N] for i in 1:N],
            res[begin+N:end-N],
            [res[end-N] for i in 1:N]
        )
    else
        sec_derivs = derivate_vector(xs, derivs; N=N)
        res = [x * d2 / d for (x, d, d2) in zip(xs, derivs, sec_derivs)] .+ 1.0
        return vcat(
            [res[begin+N+1] for i in 1:N+1],
            res[begin+N+1:end-N-1],
            [res[end-N-1] for i in 1:N+1]
        )
    end
end



"""
    mean_spectral_index(xs, ys; N::Int = 1, con = false)

Assuming that the input `ys` follow a power law distribution, 
return the mean spectral index ``\\langle S \\rangle`` of them.

The spectral index ``S`` of a generic function ``f = f(x)`` is
defined as:
```math
    S = \\frac{\\partial \\log f(x)}{\\partial \\log x} 
        = \\frac{x}{f(x)} \\frac{\\partial f(x)}{\\partial x} 
```
"""
function mean_spectral_index(xs, ys; N::Int=1, con=false)
    @assert length(xs) > 2 * N + 2 "length of xs and ys must be > 2N+2"
    vec = spectral_index(xs, ys; N=N, con=con)[begin+N+1:end-N-1]
    return sum(vec) / length(vec)
end



##########################################################################################92


"""
    power_law(x, si, b, a) ::Float64

Return the following ``y = f(x)`` "spurious" power-law value:
```math
y = f(x) = a + b \\, x^s
```
where `si` is the exponent (``s``), `b` the coefficient (``b``) and
`a` is the added constant (``a``).
"""
power_law(x, si, b, a) = a .+ b .* (x .^ si)



"""
    two_power_laws(x; switch=5.0, si_1=1.0, si_2=2.0, b=1.0, a=0.0)

Return two power laws, depending on the value of switch.
"""
function two_power_laws(x; switch=5.0, si_1=1.0, si_2=2.0, b=1.0, a=0.0)
    @assert switch > 0 "switch must be >0 !"
    if x <= switch
        return power_law(x, si_1, b, a)
    else
        return power_law(x, si_2, b / (switch^(si_2 - si_1)), a)
    end
end



function power_law_from_data(
    xs, ys,
    P0::Vector{Float64},
    fit_min::Number, fit_max::Number; con=false)

    @assert length(xs) == length(ys) "xs and ys must have same length"
    @assert length(P0) ∈ [2, 3] "length of P0 must be 2 or 3!"
    @assert min(xs...) <= fit_min "fit_min must be > min(xs...) !"
    @assert max(xs...) >= fit_max "fit_max must be < max(xs...) !"

    #p0 = abs(P0[1]) < 1.5 ? P0 : [ P0[1] - floor() , P0[2:end]...]
    p0 = P0

    mean_exp_xs = sum([log10(abs(x)) for x in xs[fit_min.<xs.<fit_max]]) / length(xs[fit_min.<xs.<fit_max])
    en_xs = 10.0^(-mean_exp_xs)
    new_xs = xs[fit_min.<xs.<fit_max] .* en_xs

    mean_exp_ys = sum([log10(abs(y)) for y in ys[fit_min.<xs.<fit_max]]) / length(ys[fit_min.<xs.<fit_max])
    en_ys = 10.0^(-mean_exp_ys)
    new_ys = ys[fit_min.<xs.<fit_max] .* en_ys

    mean_ys = sum(new_ys) / length(new_ys)
    @assert !all([isapprox(y / mean_ys, 1.0, rtol=1e-6) for y in new_ys]) "DO NOT WORK!"
    #si = mean_spectral_index(xs, ys; N=N, con=con)

    if con == false
        @assert length(p0) == 2 " si,b to be fitted, so length(p0) must be 2!"
        vec = coef(curve_fit((x, p) -> power_law(x, p[1], p[2], 0.0),
            new_xs, new_ys, p0))
        si, b, a = vcat(vec, 0.0)
        return si, b * (en_xs^si) / en_ys, a / en_ys

    else
        @assert length(p0) == 3 " si,b,a to be fitted, so length(p0) must be 3!"

        try
            fit_1 = curve_fit((x, p) -> power_law(x, p[1], p[2], p[3]),
                new_xs, new_ys, p0)
            vals_1 = coef(fit_1)
            stds_1 = stderror(fit_1)
            pers_1 = [s / v for (s, v) in zip(stds_1, vals_1)]

            si, b, a =
                if all(x -> x < 0.05, pers_1)
                        vals_1
                elseif pers_1[3] < 0.05
                        fit_2 = curve_fit((x, p) -> power_law(x, p[1], p[2], vals_1[3]),
                            new_xs, new_ys, p0)
                        vals_2 = coef(fit_2)
                        vcat(vals_2, vals_1[3])
                else
                        fit_3 = curve_fit((x, p) -> power_law(x, p[1], p[2], 0.0),
                            new_xs, new_ys, p0)
                        vals_3 = coef(fit_3)
                        fit_4 = curve_fit((x, p) -> power_law(x, vals_3[1], vals_3[2], p[3]),
                            new_xs, new_ys, p0)
                        vals_4 = coef(fit_4)

                        vcat(vals_3, vals_4)
                end

            return si, b * (en_xs^si) / en_ys, a / en_ys

        catch e
            fit_3 = curve_fit((x, p) -> power_law(x, p[1], p[2], 0.0),
                new_xs, new_ys, [p0[1], p0[2]])
            si, b, a = vcat(coef(fit_3), 0.0)

            return si, b * (en_xs^si) / en_ys, a / en_ys
        end
    end
end



function power_law_from_data(xs, ys, p0::Vector{Float64}; con=false)
    power_law_from_data(xs, ys, p0, xs[begin], xs[end]; con=con)
end


"""
    power_law_from_data(xs, ys, p0::Vector{Float64},
        fit_min::Number, fit_max::Number; con = false)

    power_law_from_data(xs, ys, p0::Vector{Float64}; con = false) = 
        power_law_from_data(xs, ys, p0, xs[begin], xs[end]; con = con)

Returns the "spurious" power-law
coefficients ``s``, ``b`` and ``a`` obtained from the fitting of the data vectors
`xs` and `ys` inside the limits `fit_min` and `fit_max`.

If `con == false`, the returned `a` is always `0.0`, because it is considered the
"pure" power-law fitting function:
```math
    y = f(x) = b \\, x^s
```
while if `con == false` it is used the spurious one:
```math
    y = f(x) = a + b \\, x^s
```
(and consequently ``a`` may be ≠0).

The fitting is performed through the function `curve_fit` of the 
[`LsqFit`](https://github.com/JuliaNLSolvers/LsqFit.jl) Julia package, which is based
on the least-squares method.
 
See also: [`power_law`](@ref)
"""
power_law_from_data


##########################################################################################92



"""
    expand_left_log(xs, ys;
        lim=1e-8, fit_min=0.05, fit_max=0.5,
        p0::Union{Vector{Float64},Nothing}=nothing,
        con::Bool=false)
"""
function expand_left_log(xs, ys;
    lim=1e-8, fit_min=0.05, fit_max=0.5,
    p0::Union{Vector{Float64},Nothing}=nothing,
    con::Bool=false)

    @assert fit_min < fit_max "fit_min must be < fit_max !"
    @assert lim < fit_min "lim must be < fit_min !"
    @assert min(xs...) <= fit_min "fit_min must be >= min(xs...) !"
    @assert max(xs...) >= fit_max "fit_max must be <= max(xs...) !"

    p_0 = isnothing(p0) ? (con == true ? [-1.0, 1.0, 0.0] : [-1.0, 1.0]) : p0
    si, b, a = power_law_from_data(
        xs, ys, p_0, fit_min, fit_max; con=con)

    i = findfirst(x -> x >= fit_min, xs) - 1
    f = xs[begin] / xs[begin+1]

    new_left_xs = unique(10 .^ range(log10(lim), log10(xs[i]), step=-log10(f)))
    new_left_ys = [power_law(x, si, b, a) for x in new_left_xs]

    if isapprox(new_left_xs[end] / xs[i], 1.0, rtol=1e-6)
        return new_left_xs, new_left_ys
    else
        return vcat(new_left_xs, xs[i]), vcat(new_left_ys, power_law(xs[i], si, b, a))
    end
end


"""
    expand_right_log(xs, ys;
        lim=3e3, fit_min=5.0, fit_max=10.0,
        p0::Union{Vector{Float64},Nothing}=nothing,
        con::Bool=false)
"""
function expand_right_log(xs, ys;
    lim=3e3, fit_min=5.0, fit_max=10.0,
    p0::Union{Vector{Float64},Nothing}=nothing,
    con::Bool=false)

    @assert fit_min < fit_max "fit_min must be < fit_max !"
    @assert lim > fit_max "lim must be > fit_max !"
    @assert min(xs...) <= fit_min "fit_min must be > min(xs...) !"
    @assert max(xs...) >= fit_max "fit_max must be < max(xs...) !"

    p_0 = isnothing(p0) ? (con == true ? [-3.0, 1.0, 0.0] : [-3.0, 1.0]) : p0
    si, b, a = power_law_from_data(
        xs, ys, p_0, fit_min, fit_max; con=con)

    i = findfirst(x -> x > fit_max, xs)
    f = xs[end] / xs[end-1]

    new_right_xs = unique(10 .^ range(log10(xs[i]), log10(lim), step=log10(f)))
    new_right_ys = [power_law(x, si, b, a) for x in new_right_xs]

    if isapprox(new_right_xs[begin] / xs[i], 1.0, rtol=1e-6)
        return new_right_xs, new_right_ys
    else
        return vcat(xs[i], new_right_xs), vcat(power_law(xs[i], si, b, a), new_right_ys)
    end
end

##########################################################################################92


"""
    expanded_IPS(ks, pks; k_in = 1e-8, k_end = 3e3, con = false)

Given the `ks` and `pks` of a chosen Power Spectrum, returns the same PS
with "longer tails", i.e. it is prolonged for higher and lower `ks` than 
the input ones.
"""
function expanded_IPS(ks, pks; k_in=1e-8, k_end=3e3,
    k1=1e-6, k2=3e-6, k3=1e1, k4=2e1)

    @assert k1 < k2 "k1 must be < k2 !"
    @assert k3 < k4 "k3 must be < k4 !"
    @assert k2 < k3 "k1-k2 and k3-k4 ranges should not overlap!"
    @assert k_in < k1 "k_in must be < k1 !"
    @assert k_end > k4 "k_end must be > k4 !"
    @assert min(ks...) <= k1 "k1 must be > min(ks...) !"
    @assert max(ks...) >= k4 "k4 must be < max(ks...) !"

    #p0_beg = con ? [1.0, 1.0, 0.0] : [1.0, 1.0]
    #p0_end = con ? [-3.0, 1.0, 0.0] : [-3.0, 1.0]
    p0_beg, p0_end = [1.0, 1.0], [-3.0, 1.0]

    new_left_ks, new_left_pks =
        k_in < ks[begin] ?
        expand_left_log(ks, pks; lim=k_in, fit_min=k1,
            fit_max=k2, p0=p0_beg, con=false) :
        (nothing, nothing)

    new_right_ks, new_right_pks =
        ks[end] < k_end ?
        expand_right_log(ks, pks; lim=k_end, fit_min=k3,
            fit_max=k4, p0=p0_end, con=false) :
        (nothing, nothing)


    new_ks, new_pks =
        if !isnothing(new_left_ks) && !isnothing(new_right_ks)
            (vcat(new_left_ks, ks[k1.<=ks.<=k4], new_right_ks),
                vcat(new_left_pks, pks[k1.<=ks.<=k4], new_right_pks))
        elseif isnothing(new_left_ks) && !isnothing(new_right_ks)
            (vcat(ks[ks.<=k4], new_right_ks),
                vcat(pks[ks.<=k4], new_right_pks))
        elseif !isnothing(new_left_ks) && isnothing(new_right_ks)
            (vcat(new_left_ks, ks[k1.<=ks]),
                vcat(new_left_pks, pks[k1.<=ks]))
        else
            (ks, pks)
        end

    return new_ks, new_pks
end


"""
    expanded_Iln(PK, l, n; N = 1024, kmin = 1e-4, kmax = 1e3, s0 = 1e-3,
        fit_left_min = 2.0, fit_left_max = 10.0, p0 = [-1.0, 1.0, 0.0], con = false)


"""
function expanded_Iln(PK, l, n; lim=1e-4, N=1024, kmin=1e-4, kmax=1e3, s0=1e-3,
    fit_left_min=2.0, fit_left_max=10.0, p0_left=nothing, con=false)

    rs, xis = xicalc(PK, l, n; N=N, kmin=kmin, kmax=kmax, r0=s0)

    p_0 = isnothing(p0_left) ? (con == true ? [-1.0, 1.0, 0.0] : [-1.0, 1.0]) : p0_left
    new_left_rs, new_left_Is = expand_left_log(rs, xis; lim=lim, fit_min=fit_left_min,
        fit_max=fit_left_max, p0=p_0, con=con)

    new_rs = vcat(new_left_rs, rs[rs.>fit_left_min])
    new_Is = vcat(new_left_Is, xis[rs.>fit_left_min])

    return new_rs, new_Is
end



##########################################################################################92



"""
    func_I04_tilde(PK, s, kmin, kmax; kwargs...)

Return the following integral:
```math
\\tilde{I}^4_0 (s) = \\int_0^\\infty \\frac{\\mathrm{d}q}{2\\pi^2} 
    q^2 \\, P(q) \\, \\frac{j_0(q s) - 1}{(q s)^4}
```
It is brute-force calcuated with `quadgk`.

## Arguments

- `PK` : function that return the Input Power Spectrum

- `s` : value of ``s`` whre the integral must be evaluated

- `kmin, kmax` : extremes (min and max) of integration

- `kwargs...` : keyword argruments that must be passed to `quadgk`,
  such as `rtol` or `atol`
"""
function func_I04_tilde(PK, s, kmin, kmax; kwargs...)
    res = quadgk(lq -> (sphericalbesselj(0, s * exp(lq)) - 1.0) * PK(exp(lq)) / (2.0 * π^2 * exp(lq)),
        log(kmin), log(kmax); kwargs...)[1]

    return res / (s^4)
end

"""
    expanded_I04_tilde(PK, ss;
        kmin=1e-6, kmax=1e3, kwargs...)
"""
function expanded_I04_tilde(PK, ss;
    kmin=1e-6, kmax=1e3, kwargs...)

    fit_1, fit_2 = 0.1, 1.0

    if all(ss .> fit_1)
        return [func_I04_tilde(PK, s, kmin, kmax; kwargs...) for s in ss]
    else
        ind = findfirst(x -> x >= fit_1, ss)
        cutted_ss = ss[ind-1:end]
        cutted_I04_tildes = [func_I04_tilde(PK, s, kmin, kmax; kwargs...) for s in cutted_ss]
        l_si, l_b, l_a = GaPSE.power_law_from_data(cutted_ss, cutted_I04_tildes,
            [-2.0, -1.0], fit_1, fit_2; con=false)
        #println("l_si, l_b, l_a = $l_si , $l_b , $l_a")
        left_I04_tildes = [GaPSE.power_law(s, l_si, l_b, l_a) for s in ss[ss.<=fit_1]]

        return vcat(left_I04_tildes, cutted_I04_tildes[2:end])
    end
end

#=
function expanded_I04_tilde(PK, ss;
    kmin = 1e-6, kmax = 1e3, kwargs...)

    fit_1, fit_2 = 0.1, 1.0
    fit_3, fit_4 = 1e3, 1e4

    if all(fit_1 .< ss .< fit_4)
        return [func_I04_tilde(PK, s, kmin, kmax; kwargs...) for s in ss]

    elseif all(ss .> fit_1)
        cutted_ss = ss[ss.<fit_4]
        cutted_I04_tildes = [func_I04_tilde(PK, s, kmin, kmax; kwargs...) for s in cutted_ss]
        r_si, r_b, r_a = GaPSE.power_law_from_data(cutted_ss, cutted_I04_tildes,
            [-4.0, -1.0, 0.0], fit_3, fit_4; con = true)
        #println("r_si, r_b, r_a = $r_si , $r_b , $r_a")
        right_I04_tildes = [GaPSE.power_law(s, r_si, r_b, r_a) for s in ss[ss.>=fit_4]]

        return vcat(cutted_I04_tildes, right_I04_tildes)

    elseif all(ss .< fit_4)
        cutted_ss = ss[ss.>fit_1]
        cutted_I04_tildes = [func_I04_tilde(PK, s, kmin, kmax; kwargs...) for s in cutted_ss]
        l_si, l_b, l_a = GaPSE.power_law_from_data(cutted_ss, cutted_I04_tildes,
            [-2.0, -1.0, 0.0], fit_1, fit_2; con = true)
        #println("l_si, l_b, l_a = $l_si , $l_b , $l_a")
        left_I04_tildes = [GaPSE.power_law(s, l_si, l_b, l_a) for s in ss[ss.<=fit_1]]

        return vcat(left_I04_tildes, cutted_I04_tildes)

    else
        cutted_ss = ss[fit_1.<ss.<fit_4]
        cutted_I04_tildes = [func_I04_tilde(PK, s, kmin, kmax; kwargs...) for s in cutted_ss]

        l_si, l_b, l_a = GaPSE.power_law_from_data(cutted_ss, cutted_I04_tildes,
            [-2.0, -1.0, 0.0], fit_1, fit_2; con = true)
        #println("l_si, l_b, l_a = $l_si , $l_b , $l_a")
        r_si, r_b, r_a = GaPSE.power_law_from_data(cutted_ss, cutted_I04_tildes,
            [-4.0, -1.0, 0.0], fit_3, fit_4; con = true)
        #println("r_si, r_b, r_a = $r_si , $r_b , $r_a")
        left_I04_tildes = [GaPSE.power_law(s, l_si, l_b, l_a) for s in ss[ss.<=fit_1]]
        right_I04_tildes = [GaPSE.power_law(s, r_si, r_b, r_a) for s in ss[ss.>=fit_4]]

        I04_tildes = vcat(left_I04_tildes, cutted_I04_tildes, right_I04_tildes)

        return I04_tildes
    end
end

=#

"""
    my_interpolation(x1, y1, x2, y2, x)
"""
function my_interpolation(x1, y1, x2, y2, x)
    @assert x1 ≤ x ≤ x2 "x1 ≤ x ≤ x2 must hold!"
    x > x1 || (return y1)
    x < x2 || (return y2)
    m = (y2 - y1) / (x2 - x1)
    q = y1 - m * x1
    y = m * x + q
    return y
end



##########################################################################################92



"""
    EPLs(
        l_si::Float64
        l_b::Float64
        l_a::Float64
        left::Float64

        spline::GaPSE.MySpline

        r_si::Float64
        r_b::Float64
        r_a::Float64
        right::Float64
    )

Contains all the information useful in order to return the value of a spline inside
the interval `left ≤ x ≤ right` and the associated power laws for the edges (with the "left"
coefficients `l_si`, `l_b` and `l_a` for `x < left` and the "right" ones `r_si`, `r_b` and 
`r_a` for `x > right`)

## Arguments 

- `l_si, l_b, l_a :: Float64` : coefficient for the spurious power-law 
  ``y = f(x) = a + b \\, x^s`` for the LEFT edge; when an input value `x < left` is
  given, the returned one is obtained from `power_law` with this coefficients (
  where, of course, `l_si` is the exponent, `l_b` the coefficient and `l_a` the 
  spurious adding constant). 

- `left::Float64` : the break between the left power-law (for `x <left`) and the 
  spline (for `x ≥ left`); its value is the `xs[begin]` one.

- `spline::GaPSE.MySpline` : spline that interpolates between the real values of the 
  integral calculated inside the range `left ≤ x ≤ right`

- `right::Float64` : the break between the right power-law (for `x ≥ left`) and the 
  spline (for `x ≤ right`); its value is the `xs[end]` one.

- `r_si, r_b, r_a :: Float64` : coefficient for the spurious power-law 
  ``y = f(x) = a + b \\, x^s`` for the RIGHT edge; when an input value `x > right` is
  given, the returned one is obtained from `power_law` with this coefficients (
  where, of course, `r_si` is the exponent, `r_b`` the coefficient and `r_a` the 
  spurious adding constant). 
  NOTE: for numerical issues, the "pure" power-law ``y = f(x) = b + x^s`` should be used. 

## Constructors

    EPLs(xs, ys, p0_left::Vector{T1}, p0_right::Vector{T2}; 
        N_left::Int = 15, N_right::Int = 15) where {T1<:Real, T2 <:Real}

- `xs` and `ys`: the input vector of values. 

- `N_left::Int = 15` : number of points to be used from the left edge for the left power law-fitting.
  It shouldn't be too low (< 4) or too high (>100).

- `N_right::Int = 15` : number of points to be used from the right edge for the right power law-fitting.
  It shouldn't be too low (< 4) or too high (>100).

- `p0_left::Vector{T1} where T1 <:Real` : vector with the initial values for the left power-law fitting; its length must
  be 2 (if you want to fit with a pure power-law ``y = f(x) = b * x^s``, so only `l_si` and `l_b` 
  are matter of concern) or 3 (if you want to fit with a spurious power-law ``y = f(x) = a + b * x^s``,
  so you are also interested in `l_a`); in the first case, the considered `l_a` will be `0.0`.
  Example: 

- `p0_right::Vector{T1} where T1 <:Real` : vector with the initial values for the right power-law fitting; its length must
  be 2 (if you want to fit with a pure power-law ``y = f(x) = b * x^s``, so only `r_si` and `r_b` 
  are matter of concern) or 3 (if you want to fit with a spurious power-law ``y = f(x) = a + b * x^s``,
  so you are also interested in `r_a`); in the first case, the considered `r_a` will be `0.0`.
  It is recommended to 

All the power-law fitting (both "pure" and spurious) are made through the 
local function `power_law_from_data`.

## Examples

```julia
julia> xs = 10 .^ range(0, 2, length=100);

julia> ys = [1.34e2 * x ^ 2.43 for x in xs];

julia> A = EPLs(xs, ys, [1.0, 1.0], [1.0, 1.0]; N_left = 10, N_right = 10)
```

See also: [`power_law_from_data`](@ref)
"""
struct EPLs
    l_si::Float64
    l_b::Float64
    l_a::Float64
    left::Float64

    spline::GaPSE.MySpline

    r_si::Float64
    r_b::Float64
    r_a::Float64
    right::Float64


    function EPLs(xs, ys, p0_left::Vector{T1}, p0_right::Vector{T2};
        N_left::Int=15, N_right::Int=15) where {T1<:Real,T2<:Real}

        @assert length(xs) == length(ys) "xs and ys must have same length!"
        @assert length(p0_left) ∈ [2, 3] "length of p0_left must be 2 or 3!"
        @assert length(p0_right) ∈ [2, 3] "length of p0_right must be 2 or 3!"
        @assert N_left > 3 "N_left must be > 3 !"
        @assert N_right > 3 "N_right must be > 3 !"
        @assert length(xs) > N_left + N_right "xs and ys are too short to be used!"

        con_left = length(p0_left) == 3 ? true : false
        con_right = length(p0_right) == 3 ? true : false

        l_si, l_b, l_a = power_law_from_data(
            xs, ys, p0_left, xs[begin+1], xs[begin+N_left]; con=con_left)

        r_si, r_b, r_a = power_law_from_data(
            xs, ys, p0_right, xs[end-1-N_right], xs[end-1]; con=con_right)

        #println("\nLEFT = $l_si , $l_b , $l_a")
        #println("RIGHT = $r_si , $r_b , $r_a")

        spline = GaPSE.MySpline(xs, ys; bc="error")

        new(l_si, l_b, l_a, xs[begin], spline, r_si, r_b, r_a, xs[end])
    end
end


"""
    (f::EPLs)(x)

Return the value of the `f::EPLs` as follows:
```math
f(x)=
\\begin{cases}
a_\\mathrm{L} + b_\\mathrm{L} \\, x ^ {s_\\mathrm{L}} \\; ,
    \\quad x < \\mathrm{left}\\\\
\\mathrm{spline}(x) \\; , \\quad \\mathrm{left} \\leq x \\leq \\mathrm{right} \\\\
a_\\mathrm{R} + b_\\mathrm{R} \\, x ^ {s_\\mathrm{R}} \\; , 
\\quad x > \\mathrm{right}
\\end{cases}
```

where ``a_\\mathrm{L}``, ``b_\\mathrm{L}``, ``s_\\mathrm{L}``, ``\\mathrm{left}``,
``\\mathrm{spline}``, ``a_\\mathrm{R}``, ``b_\\mathrm{R}``, ``s_\\mathrm{R}`` and 
``\\mathrm{right}`` are all stored inside the `EPLs` considered.

See also: [`EPLs`](@ref)
"""
function (func::EPLs)(x)
    if x < func.left
        return power_law(x, func.l_si, func.l_b, func.l_a)
    elseif x > func.right
        #warning("i am going too right! ")
        return power_law(x, func.r_si, func.r_b, func.r_a)
    else
        return func.spline(x)
    end
end


##########################################################################################92

#=

"""
    LPLs(
        l_si::Float64
        l_b::Float64
        l_a::Float64
        left::Float64

        spline::GaPSE.MySpline
    )

Contains all the information useful in order to return the value of a spline inside
the interval `left ≤ x ≤ right` and the associated power laws for the left edge (with the "left"
coefficients `l_si`, `l_b` and `l_a` for `x < left`)

## Arguments 

- `l_si, l_b, l_a :: Float64` : coefficient for the spurious power-law 
  ``y = f(x) = a + b \\, x^s`` for the LEFT edge; when an input value `x < left` is
  given, the returned one is obtained from `power_law` with this coefficients (
  where, of course, `l_si` is the exponent, `l_b` the coefficient and `l_a` the 
  spurious adding constant). 

- `left::Float64` : the break between the left power-law (for `x <left`) and the 
  spline (for `x ≥ left`); its value is the `xs[begin]` one.

- `spline::GaPSE.MySpline` : spline that interpolates between the real values of the 
  integral calculated inside the range `left ≤ x ≤ right`

## Constructors

    LPLs(xs, ys, p0_left::Vector{T1}, p0_right::Vector{T2}; 
        N_left::Int = 15, N_right::Int = 15) where {T1<:Real, T2 <:Real}

- `xs` and `ys`: the input vector of values. 

- `N_left::Int = 15` : number of points to be used from the left edge for the left power law-fitting.
  It shouldn't be too low (< 4) or too high (>100).

- `p0_left::Vector{T1} where T1 <:Real` : vector with the initial values for the left power-law fitting; its length must
  be 2 (if you want to fit with a pure power-law ``y = f(x) = b * x^s``, so only `l_si` and `l_b` 
  are matter of concern) or 3 (if you want to fit with a spurious power-law ``y = f(x) = a + b * x^s``, 
  so you are also interested in `l_a`); in the first case, the considered `l_a` will be `0.0`.
  Example: 

All the power-law fitting (both "pure" and spurious) are made through the 
local function `power_law_from_data`.

## Examples

```julia
julia> xs = 10 .^ range(0, 2, length=100);

julia> ys = [1.34e2 * x ^ 2.43 for x in xs];

julia> A = EPLs(xs, ys, [1.0, 1.0], [1.0, 1.0]; N_left = 10, N_right = 10)
```

See also: [`power_law_from_data`](@ref)
"""
struct LPLs
    l_si::Float64
    l_b::Float64
    l_a::Float64
    left::Float64

    spline::GaPSE.MySpline


    function LPLs(xs, ys, p0_left::Vector{T1}; N_left::Int=15) where {T1<:Real}

        @assert length(xs) == length(ys) "xs and ys must have same length!"
        @assert length(p0_left) ∈ [2, 3] "length of p0_left must be 2 or 3!"
        @assert N_left > 3 "N_left must be > 3 !"
        @assert length(xs) > N_left "xs and ys are too short to be used!"

        con_left = length(p0_left) == 3 ? true : false

        l_si, l_b, l_a = power_law_from_data(
            xs, ys, p0_left, xs[begin+1], xs[begin+N_left]; con=con_left)

        #println("\nLEFT = $l_si , $l_b , $l_a")
        #println("RIGHT = $r_si , $r_b , $r_a")

        spline = GaPSE.MySpline(xs, ys; bc="error")

        new(l_si, l_b, l_a, xs[begin], spline, r_si, r_b, r_a, xs[end])
    end
end


"""
    (f::EPLs)(x)

Return the value of the `f::EPLs` as follows:
```math
f(x)=
\\begin{cases}
a_\\mathrm{L} + b_\\mathrm{L} \\, x ^ {s_\\mathrm{L}} \\; ,
    \\quad x < \\mathrm{left}\\\\
\\mathrm{spline}(x) \\; , \\quad \\mathrm{left} \\leq x \\leq \\mathrm{right} \\\\
a_\\mathrm{R} + b_\\mathrm{R} \\, x ^ {s_\\mathrm{R}} \\; , 
\\quad x > \\mathrm{right}
\\end{cases}
```

where ``a_\\mathrm{L}``, ``b_\\mathrm{L}``, ``s_\\mathrm{L}``, ``\\mathrm{left}``,
``\\mathrm{spline}``, ``a_\\mathrm{R}``, ``b_\\mathrm{R}``, ``s_\\mathrm{R}`` and 
``\\mathrm{right}`` are all stored inside the `EPLs` considered.

See also: [`EPLs`](@ref)
"""
function (func::EPLs)(x)
    if x < func.left
        return power_law(x, func.l_si, func.l_b, func.l_a)
    elseif x > func.right
        #warning("i am going too right! ")
        return power_law(x, func.r_si, func.r_b, func.r_a)
    else
        return func.spline(x)
    end
end;
=#


##########################################################################################92


"""
    polynomia(x, c, b, a) ::Float64

Return the following ``y = f(x)`` 2-degree `polynomi`al value:
```math
y = f(x) = a + b \\, x + c \\, x^2
```
where `c` is the 2-degree coefficient (``c``), `b` the 1-degree coefficient (``b``) and
`a` is the added constant (``a``).
"""
polynomial(x, c, b, a) = a .+ b .* x .+ c .* x .^ 2


function curve_fit_polyn(xs, ys, known::Vector, p0::Vector)
    @assert length(known) == 3 "known must be of length 3"
    @assert length(p0) ∈ [1, 2, 3] "p0 must be of length 1,2 or 3"
    for (i, (el1, el2)) in enumerate(zip(known, p0))
        @assert typeof(el1) <: Union{Float64,Nothing} "element $i of known is of type $(typeof(el1))"
        @assert typeof(el2) == Float64 "element $i of p0 is of type $(typeof(el1))"
    end
    n_nothing = sum([isnothing(x) ? 1 : 0 for x in known])
    @assert length(p0) == n_nothing "what are you doing?"

    fit =
        if isnothing(known[1]) && !isnothing(known[2]) && !isnothing(known[3])
            curve_fit((x, p) -> polynomial(x, p[1], known[2], known[3]), xs, ys, p0)

        elseif !isnothing(known[1]) && isnothing(known[2]) && !isnothing(known[3])
            curve_fit((x, p) -> polynomial(x, known[1], p[1], known[3]), xs, ys, p0)

        elseif !isnothing(known[1]) && !isnothing(known[2]) && isnothing(known[3])
            curve_fit((x, p) -> polynomial(x, known[1], known[2], p[1]), xs, ys, p0)

        elseif isnothing(known[1]) && isnothing(known[2]) && !isnothing(known[3])
            curve_fit((x, p) -> polynomial(x, p[1], p[2], known[3]), xs, ys, p0)

        elseif isnothing(known[1]) && !isnothing(known[2]) && isnothing(known[3])
            curve_fit((x, p) -> polynomial(x, p[1], known[2], p[2]), xs, ys, p0)

        elseif !isnothing(known[1]) && isnothing(known[2]) && isnothing(known[3])
            curve_fit((x, p) -> polynomial(x, known[1], p[1], p[2]), xs, ys, p0)

        elseif isnothing(known[1]) && isnothing(known[2]) && isnothing(known[3])
            curve_fit((x, p) -> polynomial(x, p[1], p[2], p[3]), xs, ys, p0)

        else
            throw(AssertionError("how did you arrive here?"))
        end

    vals = coef(fit)
    stds = stderror(fit)
    pers = [s / v for (s, v) in zip(stds, vals)] # ratio error/value for each param

    return (vals, pers)
end

function curve_fit_polyn_onone(xs, ys, p0; en_xs=1, en_ys=1, pr::Bool = true)
    @assert length(p0) == 1 "secure"
    pr && println("1 param")
    coefs_1, sigmas_r_1 = curve_fit_polyn(xs, ys, [nothing, 0.0, 0.0], p0)
    c, b, a = vcat(coefs_1[1], 0.0, 0.0)
    return c * (en_xs^2) / en_ys, b * en_xs / en_ys, a / en_ys
end

function curve_fit_polyn_ontwo(xs, ys, p0; en_xs=1, en_ys=1, err=0.05, pr::Bool = true)
    @assert length(p0) == 2 "secure"
    try
        coefs_1, sigmas_r_1 = curve_fit_polyn(xs, ys, [nothing, nothing, 0.0], p0)

        c, b, a =
            if all(x -> abs.(x) < err, sigmas_r_1) # if all the relative errors on the parameters are < err
                pr && println("2 params, all sigmas ok")
                coefs_1[1], coefs_1[2], 0.0

            elseif abs(sigmas_r_1[2]) < err  # rel. error on "a" is too large, but on "b" is ok
                pr && println("2 params, sigma_b ok")
                coefs_2, _ = curve_fit_polyn(xs, ys, [nothing, coefs_1[2], 0.0], [coefs_1[1]])
                coefs_2[1], coefs_1[2], 0.0

            elseif abs(sigmas_r_1[1]) < err  # rel. error on "b" is too large, but on "a" is ok
                pr && println("2 params, sigma_c ok")
                coefs_2, _ = curve_fit_polyn(xs, ys, [coefs_1[1], nothing, 0.0], [coefs_1[2]])
                coefs_1[1], coefs_2[1], 0.0

            else
                pr && println("2 params, both sigmas too high")
                coefs_3, _ = curve_fit_polyn(xs, ys, [nothing, 0.0, 0.0], [coefs_1[1]])
                coefs_4, _ = curve_fit_polyn(xs, ys, [coefs_3[1], nothing, 0.0], [coefs_1[2]])

                coefs_3[1], coefs_4[1], 0.0
            end

        return c * (en_xs^2) / en_ys, b * en_xs / en_ys, a / en_ys

    catch e
        if isa(AssertionError, e)
            throw(e)
        else
            println(
                "WARNING: there was an exception, so I am trying to fit with one parameter. " *
                "Try to change the input starting points."
            )
            return curve_fit_polyn_onone(xs, ys, [p0[1]]; en_xs=en_xs, en_ys=en_ys, pr=pr)
        end
    end
end

function curve_fit_polyn_onthree(xs, ys, p0; en_xs=1, en_ys=1, err=0.05, pr::Bool = true)
    try
        coefs_1, sigmas_r_1 = curve_fit_polyn(xs, ys, [nothing, nothing, nothing], p0)

        c, b, a =
            if all(x -> abs.(x) < err, sigmas_r_1) # if all the relative errors on the parameters are < err
                pr && println("3 params, all sigmas_r ok")
                coefs_1

            elseif abs(sigmas_r_1[1]) < err && abs(sigmas_r_1[2]) < err # rel. errors on "c" and "b" ok
                pr && println("3 params, sigma_c and sigma_b ok")
                coefs_2, _ = curve_fit_polyn(xs, ys, [coefs_1[1], coefs_1[2], nothing], [coefs_1[3]])
                coefs_1[1], coefs_1[2], coefs_2[1]

            elseif abs(sigmas_r_1[1]) < err && abs(sigmas_r_1[3]) < err # rel. errors on "c" and "a" ok
                pr && println("3 params, sigma_c and sigma_a ok")
                coefs_2, _ = curve_fit_polyn(xs, ys, [coefs_1[1], nothing, coefs_1[3]], [coefs_1[2]])
                coefs_1[1], coefs_2[1], coefs_1[3]

            elseif abs(sigmas_r_1[2]) < err && abs(sigmas_r_1[3]) < err # rel. errors on "b" and "a" ok
                pr && println("3 params, sigma_b and sigma_a ok")
                coefs_2, _ = curve_fit_polyn(xs, ys, [nothing, coefs_1[2], coefs_1[3]], [coefs_1[1]])
                coefs_2[1], coefs_1[2], coefs_1[3]

            elseif abs(sigmas_r_1[1]) < err  # rel. errors on "b" and "c" are too large, but on "a" is ok
                pr && println("3 params, only sigma_c ok")
                coefs_2, sigmas_r_2 = curve_fit_polyn(xs, ys, [coefs_1[1], nothing, nothing], [coefs_1[2], coefs_1[3]])

                if all(x -> abs.(x) < err, sigmas_r_2)
                        coefs_1[1], coefs_2[1], coefs_2[2]
                elseif abs(sigmas_r_2[1]) < err
                        coefs_3, _ = curve_fit_polyn(xs, ys, [coefs_1[1], coefs_2[1], nothing], [coefs_2[2]])
                        coefs_1[1], coefs_2[1], coefs_3[1]
                else
                        coefs_3, _ = curve_fit_polyn(xs, ys, [coefs_1[1], nothing, coefs_2[2]], [coefs_2[1]])
                        coefs_1[1], coefs_3[1], coefs_2[2]
                end

            elseif abs(sigmas_r_1[2]) < err  # rel. errors on "a" and "c" are too large, but on "b" is ok
                pr && println("3 params, only sigma_b ok")
                coefs_2, sigmas_r_2 = curve_fit_polyn(xs, ys, [nothing, coefs_1[2], nothing], [coefs_1[1], coefs_1[3]])

                if all(x -> abs.(x) < err, sigmas_r_2)
                        coefs_2[1], coefs_1[2], coefs_2[2]
                elseif abs(sigmas_r_2[1]) < err
                        coefs_3, _ = curve_fit_polyn(xs, ys, [coefs_2[1], coefs_1[2], nothing], [coefs_2[2]])
                        coefs_2[1], coefs_1[2], coefs_3[1]
                else
                        coefs_3, _ = curve_fit_polyn(xs, ys, [nothing, coefs_1[2], coefs_2[2]], [coefs_2[1]])
                        coefs_3[1], coefs_1[2], coefs_2[2]
                end

            elseif abs(sigmas_r_1[3]) < err  # rel. errors on "a" and "b" are too large, but on "c" is ok
                pr && println("3 params, only sigma_a ok")
                coefs_2, sigmas_r_2 = curve_fit_polyn(xs, ys, [nothing, nothing, coefs_1[3]], [coefs_1[1], coefs_1[2]])

                if all(x -> abs.(x) < err, sigmas_r_2)
                        coefs_2[1], coefs_2[2], coefs_1[3]
                elseif abs(sigmas_r_2[1]) < err
                        coefs_3, _ = curve_fit_polyn(xs, ys, [coefs_2[1], nothing, coefs_1[3]], [coefs_2[2]])
                        coefs_2[1], coefs_3[2], coefs_1[3]
                else
                        coefs_3, _ = curve_fit_polyn(xs, ys, [nothing, coefs_1[2], coefs_1[3]], [coefs_2[1]])
                        coefs_3[1], coefs_2[2], coefs_3[1]
                end

            else
                pr && println("3 params, none sigma ok")
                coefs_2, _ = curve_fit_polyn(xs, ys, [nothing, 0.0, 0.0], [coefs_1[1]])
                coefs_3, _ = curve_fit_polyn(xs, ys, [coefs_2[1], nothing, 0.0], [coefs_1[2]])
                coefs_4, _ = curve_fit_polyn(xs, ys, [coefs_2[1], coefs_3[1], nothing], [coefs_1[3]])
                coefs_2[1], coefs_3[1], coefs_4[1]
            end

        return c * (en_xs^2) / en_ys, b * en_xs / en_ys, a / en_ys

    catch e
        if e==BoundsError
            e
        elseif isa(AssertionError, e)
            throw(e)
        else
            println(
                "WARNING: there was an exception, so I am trying to fit with two parameters. " *
                "Try to change the input starting points."
            )
            return curve_fit_polyn_ontwo(xs, ys, [p0[1], p0[2]]; en_xs=en_xs, en_ys=en_ys, pr=pr)
        end
    end
end

function polynomial_from_data(
    xs, ys,
    P0::Vector{Float64},
    fit_min::Number, fit_max::Number; err::Float64=0.05, pr::Bool = true)

    @assert length(xs) == length(ys) "xs and ys must have same length"
    @assert length(P0) ∈ [1, 2, 3] "length of P0 must be 1, 2 or 3!"
    @assert min(xs...) <= fit_min "fit_min must be > min(xs...) !"
    @assert max(xs...) >= fit_max "fit_max must be < max(xs...) !"
    @assert 0 < err < 1 "the relative error \"err\" on the params must be 0<err<1"

    #p0 = abs(P0[1]) < 1.5 ? P0 : [ P0[1] - floor() , P0[2:end]...]
    p0 = P0

    mean_exp_xs = sum([log10(abs(x)) for x in xs[fit_min.<xs.<fit_max]]) / length(xs[fit_min.<xs.<fit_max])
    en_xs = 10.0^(-mean_exp_xs)
    new_xs = xs[fit_min.<xs.<fit_max] .* en_xs

    mean_exp_ys = sum([log10(abs(y)) for y in ys[fit_min.<xs.<fit_max]]) / length(ys[fit_min.<xs.<fit_max])
    en_ys = 10.0^(-mean_exp_ys)
    new_ys = ys[fit_min.<xs.<fit_max] .* en_ys

    mean_ys = sum(new_ys) / length(new_ys)
    @assert !all([isapprox(y / mean_ys, 1.0, rtol=1e-6) for y in new_ys]) "DO NOT WORK!"
    #si = mean_spectral_index(xs, ys; N=N, con=con)

    if length(p0) == 1
        return curve_fit_polyn_onone(new_xs, new_ys, p0; en_xs=en_xs, en_ys=en_ys, pr=pr)
    elseif length(p0) == 2
        return curve_fit_polyn_ontwo(new_xs, new_ys, p0; en_xs=en_xs, en_ys=en_ys, err=err, pr=pr)
    else
        return curve_fit_polyn_onthree(new_xs, new_ys, p0; en_xs=en_xs, en_ys=en_ys, err=err, pr=pr)
    end
end



function polynomial_from_data(xs, ys, p0::Vector{Float64}; kwargs...)
     polynomial_from_data(xs, ys, p0, xs[begin], xs[end]; kwargs...)
end


"""
    polynomial_from_data(xs, ys, p0::Vector{Float64},
        fit_min::Number, fit_max::Number; con = false)

    polynomial_from_data(xs, ys, p0::Vector{Float64}; con = false) = 
        polynomial_from_data(xs, ys, p0, xs[begin], xs[end]; con = con)

Returns the 2-degree polynomial
coefficients ``c``, ``b`` and ``a`` obtained from the fitting of the data vectors
`xs` and `ys` inside the limits `fit_min` and `fit_max`.

The fitting function is always a second-degree polynomio, but depeding on
the starting values vector `p0` can assume three different forms:

- if `length(p0) == 3`, the fitting polynomio is
  ```math
    f(x) = a + b \\, x + c \\, x^2
  ```

- if `length(p0) == 2`, the fitting polynomio is
  ```math
    f(x) = b \\, x + c \\, x^2
  ```

- if `length(p0) == 1`, the fitting polynomio is
  ```math
    f(x) = c \\, x^2
  ```


The fitting is performed through the function `curve_fit` of the 
[`LsqFit`](https://github.com/JuliaNLSolvers/LsqFit.jl) Julia package, which is based
on the least-squares method.
 
See also: [`polynomial`](@ref)
"""
polynomial_from_data
