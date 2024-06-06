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

#=
"""
    ξ_from_PS(f_in;
        int_k_min::Float64 = 1e-3, int_k_max::Float64 = 1e1,
        L::Int = 0, N::Int = 1024, pr::Bool = true,
        s0::Union{Nothing,Float64} = nothing,
        right::Union{Float64,Nothing} = nothing)

TBW
"""
function ξ_from_PS(f_in;
    int_k_min::Float64 = 1e-3, int_k_max::Float64 = 1e1,
    L::Int = 0, N::Int = 1024, pr::Bool = true,
    s0::Union{Nothing,Float64} = nothing,
    right::Union{Float64,Nothing} = nothing)

    s_0 = isnothing(s0) ? 1.0 / int_k_max : s0

    t1 = time()
    ss, xis = xicalc(k -> f_in(k), L, 0;
        N = N, kmin = int_k_min, kmax = int_k_max, r0 = s_0)
    t2 = time()
    pr && println("\ntime needed for this TPCF computation [in s] = $(t2-t1)\n")

    if isnothing(right)
        return ss, xis
    else
        return ss[ss .< right], xis[ss .< right]
    end
end


function ξ_from_PS(ks, pks;
    int_k_min::Float64 = 1e-3, int_k_max::Float64 = 1e1,
    epl::Bool = true,
    N_left::Int = 12, N_right::Int = 12,
    p0_left = [1.0, 1.0], p0_right = [-2.0, 1.0],
    kwargs...)

    @assert length(ks) == length(pks) "xs and ys must have same length"
    s_0 = 1.0 / max(ks...)
    right = 1.0 / min(ks...)

    f_in, INT_k_min, INT_k_max =
        if epl == true
            if all(pks[begin:begin+5] .≈ 0.0) && all(pks[end-5:end] .≈ 0.0)
                spl = GaPSE.MySpline(ks, pks; bc="error")
                f(k) = ((k ≤ ks[1]) || (k ≥ ks[end])) ? 0.0 : spl(k)
                f, int_k_min, int_k_max
            else
                EPLs(ks, pks, p0_left, p0_right;
                        N_left = N_left, N_right = N_right), int_k_min, int_k_max
            end
        else
            GaPSE.MySpline(ks, pks; bc = "error"), min(ks...), max(ks...)
        end

    return ξ_from_PS(f_in; int_k_min = INT_k_min, int_k_max = INT_k_max,
        s0 = s_0, right = right, kwargs...)
end


function ξ_from_PS(input::String; comments = false, kwargs...)

    pk_table = readdlm(input, comments=comments)
    ks = convert(Vector{Float64}, pk_table[:, 1])
    pks = convert(Vector{Float64}, pk_table[:, 2])

    return ξ_from_PS(ks, pks; kwargs...)
end


##########################################################################################92


function print_ξ_from_PS(input::String, out::String;
    L::Int = 0, N::Int = 1024,
    pr::Bool = true, kwargs...)

    check_parent_directory(out)
    check_namefile(out)

    pr && println("""\nI'm computing the TPCF from the file "$input" """)

    time_1 = time()
    vec = ξ_from_PS(input; N = N, L = L, pr = pr, kwargs...)
    time_2 = time()

    isfile(out) && run(`rm $out`)
    open(out, "w") do io
        println(io, BRAND)

        println(io, "# Two-Point Correlation Function computation of the file: \"$input\"")
        println(io, "#\n# For this TPCF computation we set: ")
        println(io, "# \t #points used in Fourier transform N = $N")
        println(io, "# \t multipole degree in consideration L = $L")
        println(io, "# computational time needed (in s) : $(@sprintf("%.4f", time_2-time_1))")
        print(io, "# kwards passed to \"print_ξ_from_PS\": ")

        println(io, "\n# \t\tL = $L")
        if !isempty(kwargs)
            for key in keys(kwargs)
                println(io, "# \t\t$(key) = $(kwargs[key])")
            end
        end
        println(io, "# ")
        println(io, "# s [Mpc/h_0] \t \t  xi")
        for (s, xis) in zip(vec[1], vec[2])
            println(io, "$s \t $xis")
        end
    end
end

function print_ξ_from_PS(ks, pks, out::String;
    L::Int=0, N::Int=1024,
    pr::Bool=true, kwargs...)

    check_parent_directory(out)
    check_namefile(out)

    pr && println("""\nI'm computing the TPCF from the two input vectors.""")

    time_1 = time()
    vec = ξ_from_PS(ks, pks; N=N, L=L, pr=pr, kwargs...)
    time_2 = time()

    isfile(out) && run(`rm $out`)
    open(out, "w") do io
        println(io, BRAND)

        println(io, "# Two-Point Correlation Function computation from two input vectors.")
        println(io, "#\n# For this PS_multipole computation we set: ")
        println(io, "# \t #points used in Fourier transform N = $N")
        println(io, "# \t multipole degree in consideration L = $L")
        println(io, "# computational time needed (in s) : $(@sprintf("%.4f", time_2-time_1))")
        print(io, "# kwards passed to \"print_ξ_from_PS\": ")

        println(io, "\n# \t\tL = $L")
        if !isempty(kwargs)
            for key in keys(kwargs)
                println(io, "# \t\t$(key) = $(kwargs[key])")
            end
        end
        println(io, "# ")
        println(io, "# s [Mpc/h_0] \t \t  xi")
        for (s, xis) in zip(vec[1], vec[2])
            println(io, "$s \t $xis")
        end
    end
end


##########################################################################################92

=#

