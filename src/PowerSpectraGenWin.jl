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
    GenericWindow(
        comdist::Vector{Float64}
        multipoles::Vector{Vector{Float64}}
        splines::Vector{GaPSE.MySpline}
    )

Stores the multipoles of a generic window function, computed as:
```math
    Q_{\\ell_1} = \\int_0^{\\infty} \\mathrm{d}s_1 \\, s_1^2 \\, \\phi(s_1) \\, F_{\\ell_1}(s_1,s)
```

with some FFT algorithm. See Eq. (2.13) of Castorina, Di Dio (2021) for more details.

## Constructors

    GenericWindow(file::String; comments::Bool=true, xdt::DataType=Float64, ydt::DataType=Float64)

Read the file `file` and create a `GenericWindow` struct having
- as first element the comoving distances stored in the first column (with the input type `xdt`)
- as second element a vector that contains all the following columns 
  (with the input type `ydt`), which are the multipoles `L=0,1,2,...` of the Window FUnction considered
- as second element a vector that contains all the splines of that multipoles

If the file start with comments (lines starting with #), set `comments = true`.
"""
struct GenericWindow
    comdist::Vector{Float64}
    multipoles::Vector{Vector{Float64}}
    splines::Vector{GaPSE.MySpline}

    function GenericWindow(file::String; comments::Bool=true, xdt::DataType=Float64, ydt::DataType=Float64)

        table = readdlm(file, comments=comments)
        ss = vecstring_to_vecnumbers(table[:, 1]; dt=xdt)
        all = [vecstring_to_vecnumbers(col; dt=ydt)
                for col in eachcol(table[:, 2:end])]

        for col in all
            @assert length(ss) == length(col) "the columns must have all the same length"
        end

        splines = [GaPSE.MySpline(ss, col; bc="error") for col in all]

        new(ss, all, splines)
    end
end



"""
    create_file_for_XiMultipoles(out::String, names::Vector{String}, 
        effect::Union{String, Integer}, group::String="GNC"; 
        comments::Bool=true, xdt::DataType=Float64, ydt::DataType=Float64)

Read the column number `effect` (if is an Integer) or the one corresponding to the GR effect `effect`
for the input group `group` (if is a String) from all the filenames stored in the Vector `names`, 
and save them in a file named `out`.

The first column of `out` will be the same as the first column of the first filename in `names`; it
is however checked internally if the first column of all the other files coincides with this one.
The following columns of `out` follow the order in `names`. Note that `effect`, if passed as Integer,
must be > 1 (because 1 is the index of the first column, used as x-axis).

`group` must be one among the following: $(GaPSE.VALID_GROUPS)
If `group=$(GaPSE.VALID_GROUPS[end])`, then `effect` must be an integer (because 
you are not selecting a specific effect in one of the native GaPSE groups).

`xdt` and `ydt` are the data types to be used for respectively the first column and the 2-3-4-... columns 
of `out`.
Set `comments=true` if the files in `names` start with a header that must be skipped (its lines must start with
#, otherwise they will not be recognised as comments).

## Example

julia> run(`cat file_1.txt`)
# Generic comment line
# of the file_1.txt
1.0  0.999999  0.34545   0.00991
...  ...       ...       ...

julia> run(`cat file_2.txt`)
# same, for file_2.txt
1.0  0.58244  0.12123    0.000154
...  ...       ...       ...

julia> create_file_for_XiMultipoles("mix.txt", ["file_1.txt", "file_2.txt"], 3, "generic");
julia> run(`cat mix.txt`)
###############
#    GaPSE    #
############### 
#
#
# This is a table containing the multipoles of the Two-Point Correlation Function (TPCF) 
# for a generic group effect [not given, provied only the index 2] taken from the files:
#   - L = 0 : file_1.txt
#   - L = 1 : file_2.txt
#
# s [Mpc/h_0]     xi_{L=0}      xi_{L=1}          
1.0   0.34545   0.12123 
...   ...       ...
"""
function create_file_for_XiMultipoles(out::String, names::Vector{String},
    effect::Union{String,Integer}, group::String="GNC";
    comments::Bool=true, xdt::DataType=Float64, ydt::DataType=Float64)

    @assert length(names) > 0 "at least one file name must be given!"
    @assert group ∈ VALID_GROUPS "group must be one among $(VALID_GROUPS); $group is not valid!"

    if typeof(effect) <: Integer
        @assert effect > 1 "if you pass effect as number, it must" *
                            " be an integer > 1, not $effect !"

    elseif group == "GNC"
        error = "$effect is not a valid GR effect name.\n" *
                "Valid GR effect names are the following:\n" *
                string(GaPSE.GR_EFFECTS_GNC .* " , "...)
        @assert (effect ∈ GaPSE.GR_EFFECTS_GNC) error

    elseif group == "LD"
        error = "$effect is not a valid GR effect name.\n" *
                "Valid GR effect names are the following:\n" *
                string(GaPSE.GR_EFFECTS_LD .* " , "...)
        @assert (effect ∈ GaPSE.GR_EFFECTS_LD) error

    elseif group == "GNCxLD"
        error = "$effect is not a valid GR effect name.\n" *
                "Valid GR effect names are the following:\n" *
                string(GaPSE.GR_EFFECTS_GNCxLD .* " , "...)
        @assert (effect ∈ GaPSE.GR_EFFECTS_GNCxLD) error

    elseif group == "LDxGNC"
        error = "$effect is not a valid GR effect name.\n" *
                "Valid GR effect names are the following:\n" *
                string(GaPSE.GR_EFFECTS_LDxGNC .* " , "...)
        @assert (effect ∈ GaPSE.GR_EFFECTS_LDxGNC) error

    elseif group == "generic"
        throw(AssertionError(
            """if you set group="generic", you should't give in input a string (because 
            you are not selecting a specific effect in a group) but an integer for effect."""
        ))
    else
        throw(AssertionError("how did you arrive here?"))
    end


    index = (typeof(effect) <: Integer || group == "generic") ? begin
        effect
    end : (group == "GNC") ? begin
        GaPSE.INDEX_GR_EFFECT_GNC[effect] + 2
    end : (group == "LD") ? begin
        GaPSE.INDEX_GR_EFFECT_LD[effect] + 2
    end : (group == "GNCxLD") ? begin
        GaPSE.INDEX_GR_EFFECT_GNCxLD[effect] + 2
    end : (group == "LDxGNC") ? begin
        GaPSE.INDEX_GR_EFFECT_LDxGNC[effect] + 2
    end : throw(AssertionError("how did you arrive here?"))

    table = readdlm(names[1], comments=comments)
    ss = GaPSE.vecstring_to_vecnumbers(table[:, 1]; dt=xdt)

    ALL = [
        begin
            ops = readdlm(name, comments=comments)
            here_ss = GaPSE.vecstring_to_vecnumbers(ops[:, 1]; dt=xdt)
            col = GaPSE.vecstring_to_vecnumbers(ops[:, index]; dt=ydt)
            @assert length(ss) == length(col) "the columns must have all the same length, $name differs!"
            @assert all(ss .≈ here_ss) "the ss must have all the same values, $name differs!"
            col
        end for name in names]

    open(out, "w") do io
        println(io, GaPSE.BRAND)
        println(io, "#\n# This is a table containing the multipoles of the Two-Point Correlation Function (TPCF) ")

        EFFECT = typeof(effect) == String ? effect : "[not given, provied only the index $effect]"
        if group == "GNC"
            println(io, "# for the Galaxy Number Counts GR effect $EFFECT" *
                        "\n#  taken from the files:")
        elseif group == "LD"
            println(io, "# for the Luminosity Distance perturbations GR effect $EFFECT" *
                        "\n# taken from the files:")
        elseif group == "GNCxLD"
            println(io, "# for the cross correlations between " *
                        "Galaxy Number Counts and Luminosity Distance perturbations \n#" *
                        "effect $EFFECT taken from the files:")
        elseif group == "LDxGNC"
            println(io, "# for the cross correlations between " *
                        "Luminosity Distance perturbations and Galaxy Number Counts \n#" *
                        "effect $EFFECT taken from the files:")
        elseif group == "generic"
            println(io, "# for a generic group " *
                        "effect $EFFECT taken from the files:")
        else
            throw(AssertionError("how did you arrive here?"))
        end

        for (i, name) in enumerate(names)
            println(io, "#   - L = $(i-1) : $name")
        end

        println(io, "#\n# s [Mpc/h_0] \t" .* join(["xi_{L=$(i-1)} \t " for i in 1:length(names)]))
        for (i, s) in enumerate(ss)
            println(io, "$s \t " *
                        join([" $(v[i]) \t " for v in ALL]))
        end
    end
end



"""
    XiMultipoles(
        comdist::Vector{Float64}
        multipoles::Vector{Vector{Float64}}
    )

Stores the multipoles of a generic Two-Point Correlation Function.

## Constructors

    XiMultipoles(file::String; comments::Bool=true, xdt::DataType=Float64, ydt::DataType=Float64)

Read the file `file` and create a `XiMultipoles` struct having
- as first element the comoving distances stored in the first column (with the input type `xdt`)
- as second element a vector that contains all the following columns 
  (with the input type `ydt`), which are the multipoles `L=0,1,2,...` of the TPCF considered

If the file start with comments (lines starting with #), set `comments = true`.
"""
struct XiMultipoles
    comdist::Vector{Float64}
    multipoles::Vector{Vector{Float64}}

    function XiMultipoles(file::String; comments::Bool=true, xdt::DataType=Float64, ydt::DataType=Float64)

        table = readdlm(file, comments=comments)
        ss = vecstring_to_vecnumbers(table[:, 1]; dt=xdt)
        all = [vecstring_to_vecnumbers(col; dt=ydt)
                for col in eachcol(table[:, 2:end])]

        for col in all
            @assert length(ss) == length(col) "the columns must have all the same length"
        end

        new(ss, all)
    end
end





##########################################################################################92




"""
    PS_multipole_GenWin(
        ximult::Union{XiMultipoles,String}, genwin::Union{GenericWindow,String};
        alg::Symbol=:fftlog, L::Int=0,
        cut_first_n::Int=0, cut_last_n::Int=0,
        kwargs...)

Return the `L`-order Power Spectrum (PS) multipole for a generic window function, through the
following Fast Fourier Transform and the effective redshift approximation:

```math
    \\left\\langle \\hat{P}_L(k) \\right\\rangle = 
        \\frac{2 L + 1}{A} (-i)^L
        \\sum_{\\ell=0}^{\\infty} 
        \\sum_{\\ell_1=0}^{\\infty} 
        \\begin{pmatrix}
            L & \\ell & \\ell_1 \\\\
            0 & 0 & 0
        \\end{pmatrix}^2
        \\int_0^{\\infty}\\!\\mathrm{d} s \\, s^2 \\, \\xi_\\ell(s, s_{\\rm eff}) \\, 
        j_L(k s) \\, Q_{\\ell_1}(s) \\, ,
```

where:
- ``\\left\\langle \\hat{P}_L(k) \\right\\rangle`` is the order ``L`` Power Spectrum of the effect we 
  are interested in; we are basing this expression on the Yamamoto estimator (see Yamamoto (2000) and 
  Yamamoto (2006))
- ``A`` is a normalization constant
- the 2x3 matrix represents the Wigner-3j symbols
- ``\\xi_\\ell`` is the order ``\\ell`` multipole of the Two-Point Correlation Function (TPCF)
- ``j_L`` is the spherical Bessel function of order ``L``
- ``s_{\\mathrm{eff}}`` is the comoving distance associated with the effective redshift (see the 
  `TUTORIAL.ipynb` notebook)

``Q_{\\ell_1}`` can be easily estimated with FFT methods:

```math
\\begin{split}
    Q_{\\ell_1}(s) &:= \\int_0^{\\infty}\\mathrm{d}s_1 \\; s_1^2 \\;
    \\phi(s_1) \\; F_{\\ell_1}(s_1, s) \\\\
    F_{\\ell_1} (s_1 , s) &:= 
    \\int_{4\\pi} \\mathrm{d}\\Omega_{\\mathbf{\\hat{s}}} \\,  
    \\int_{4\\pi} \\mathrm{d}\\Omega_{\\mathbf{\\hat{s}}_1} \\,
    \\phi(s_2) \\, W(\\mathbf{\\hat{s}}_1) \\, W(\\mathbf{\\hat{s}}_2) \\,
    \\mathcal{L}_{\\ell_1}(\\mathbf{\\hat{s}} \\cdot \\mathbf{\\hat{s}}_1)  \\, .
\\end{split}
```

where:
- ``\\mathcal{L}_{\\ell_1}`` is the Legendre polynomial of order ``\\ell_1``
- ``\\mathrm{d}\\Omega_{\\mathbf{\\hat{s}}}`` is the infinitesimal solid angle pointing in the 
  direction of the versor ``\\mathbf{\\hat{s}}``
- ``\\phi(s)`` and  ``W(\\mathbf{\\hat{s}})`` are respectively the radial and angular part of your 
  window function (we remember that we assumed that such separability of the window function is possible)

Check Eq.(2.13) of Castorina, Di Dio for the theoretical explanation of this formula.
     
Currenlty, there are two algorithms you can choose in order to perform the computation; you can choose 
which one to use through the keyword value `alg`:
- `alg = :fftlog` (default and recommended option) will employ the [FFTLog](https://github.com/marcobonici/FFTLog.jl) 
  algorithm.
- `alg = :twofast` will employ the TwoFAST `xicalc` function of the [TwoFAST](https://github.com/hsgg/TwoFAST.jl) 
  Julia package. Note that in the computation the integration range ``0\\leq s \\leq \\infty`` 
  is reduced to `int_s_min ≤ s ≤ int_s_max`. This alogrithm is not the ideal choise, because TwoFAST is conceived
  for the direction PS -> TPCF, while is not 100% trustworthy for the other way round.

IMPORTANT: no matter which algorithm you choose, you will need to give the input data in a
LOGARITHMICALLY DISTRIBUTED scale. A linear distribution does not fit for the algorithms to apply.


## Inputs

- `ximult::Union{XiMultipoles,String}` : Two-Point Correlation functions to be used in the computation. You can
  provide either a `XiMultipoles` struct containing them or the String filename where they are stored (that will 
  be internally open with `XiMultipoles` too). 
  The file must have a structure analogous to the `genwin` one (see the next Inputs item). You can use 
  `create_file_for_XiMultipoles` to produce such a file.

- `genwin::Union{GenericWindow,String}` : multipoles ``Q_{\\ell_1}`` of the generic window function you want 
  to consider. You can provide either a `GenericWindow` struct containing them or the String filename 
  where they are stored (that will be internally open with `GenericWindow` too). 
  The file must have the following structure
  ```text
    \$ cat Ql_multipoles.txt 
    # Any comment line in the file must start with a #
    # you can have how many comment lines you want in the header; they 
    # will be all skipped.
    # Then you must provide in blank space separated columns:
    # - as first column, the comoving distance values, measured in Mpc/h_0
    # - from the second column onwards, all the Q_{\ell_1} multipoles you want;
    #   they must be ordered followinf the ascending multipole order (so \ell_1 = 0
    #   must be the 2 column), and you can go as further as you want in the multipole
    #   order
    # 
    # s [Mpc/h_0]      Q_{l1=0}      Q_{l1=1}      Q_{l1=2}      ...
    1.0     0.9999999999999999      1.445269850978461e-7      0.000011917268941324522    ...
    21.0    0.9832914433168294      -0.0025537781362117177  -0.0033199998619560947        ...
    41.0    0.9669175943095181      -0.004923364937797496      -0.006463561496567318        ...     
    ...     ...                  ...                      ...
  ```


##  Optional arguments

Depending on the algorithm you choose, the options would change. The options in common are:
- `pr::Bool=true` : want to print the automatic messages to the screen?
- `L::Int=0` : which multipole order should I use for this computation? IT MUST MATCH 
  THE MULTIPOLE ORDER OF THE INPUT TPCF!
- `cut_first_n::Int=0` and `cut_last_n::Int=0` : you can cout the first and/or last n elements
  of the input data, if they are highly irregular.

The specific ones for `alg = :fftlog` are:
- `ν::Union{Float64,Nothing} = nothing` : bias parameter, i.e. exponent used to "balance" the curve;
  if `nothing`, will be set automatically to `1.5`
- `n_extrap_low::Int = 500` and `n_extrap_high::Int = 500` : number of points to concatenate on the left/right
  of the input x-axis `ss` vector, logarithmically distributed with the same ratio of the left/right-edge
  elements of `ss`.
- `n_pad::Int = 500` : number of zeros to be concatenated both on the left and
  on the right of the input function. They stabilize a lot the algorithm.

The specific ones for `alg = :twofast` are:
- `epl::Bool=true` : do you want to extend the edges of the input vectors using two fitted
  power-laws (obtained from `EPLs`)
- `N_left::Int = 12` and `N_right::Int = 12` : number of points from left
  right edges to be used for the power law fitting in `EPLs`. They matters only
  if in the given input file ξ is not defined until the extremes of integration
  `int_s_min` and `int_s_max`.
- `int_s_min::Float64 = 1e-1` and `int_s_max::Float64 = 1e3`: extremes of integration; if `epl`
  is set to `false`, their values will be automatically set to `min(ss...)` and `max(ss...)`
  respectively. Their values do matter only if `epl=true`. 
- `p0_left=[-2.0, 1.0]` and `p0_right=[-2.0, 1.0]`: vectors with the initial values for the left/right 
  power-law fitting of `EPLs`; the power-law is in the form ``y = f(x) = b * x^s``, so the first vector 
  value is the initial value of ``s`` (and of course the second is the one of ``b``).
- `k0::Union{Nothing,Float64} = nothing` : starting point for the `xicalc` function; if `nothing`, 
  it will be set `k0 = 1.0 / max(ss...)`
- `right::Union{Float64,Nothing} = nothing` : do you want to cut the output elements with 
  `ks .> right`? if set to `nothing`, no cut will be done.
- `N::Int = 1024` : number of points to be used in Fourier transform 

## Returns

A `Tuple{Vector{Float64}, Vector{Float64}}` with:
- the `k` values vector as first element;
- the correspoding PS `pk` values vector as second one.


See also: [`create_file_for_XiMultipoles`](@ref), [`XiMultipoles`](@ref), 
[`GenericWindow`](@ref),
[`V_survey`](@ref), [`A`](@ref), [`A_prime`](@ref),
[`EPLs`](@ref),  [`print_PS_multipole`](@ref)
"""
function PS_multipole_GenWin(
    ximult::Union{XiMultipoles,String}, genwin::Union{GenericWindow,String};
    alg::Symbol=:fftlog, L::Int=0,
    cut_first_n::Int=0, cut_last_n::Int=0,
    kwargs...)

    @assert cut_first_n ≥ 0 "cut_first_n ≥ 0 must hold!"
    @assert cut_last_n ≥ 0 "cut_last_n ≥ 0 must hold!"

    XIMULT = typeof(ximult) == String ? XiMultipoles(ximult) : ximult
    GENWIN = typeof(genwin) == String ? GenericWindow(genwin) : genwin

    @assert cut_first_n + cut_last_n < length(XIMULT.comdist) "cut_first_n + cut_last_n < length(XIMULT.comdist) must hold!"

    a, b = 1 + cut_first_n, length(XIMULT.comdist) - cut_last_n
    SS, VEC_FS = XIMULT.comdist[a:b], [vec[a:b] for vec in XIMULT.multipoles]

    if alg == :twofast

        ks, _ = GaPSE.TwoFAST_PS_multipole(SS, VEC_FS[1]; cut_first_n=0, cut_last_n=0, kwargs...)
        res = zeros(length(ks))

        #=
        for l in 0:length(XIMULT.multipoles)-1, l1 in 0:length(GENWIN.splines)-1

            w3j = convert(Float64, wigner3j(L, l, l1, 0, 0, 0))

            if (w3j ≈ 0.0) == false
                qls = [GENWIN.splines[l1+1](s) for s in SS]
                _, pks = GaPSE.TwoFAST_PS_multipole(SS, VEC_FS[l+1] .* qls; L=L, cut_first_n=0, cut_last_n=0, kwargs...)
                res += w3j^2 .* (2.0 * l1 + 1.0) .* VEC_FS[l+1] .* qls
                #s_index = findfirst(x->x>1000.0, SS)
                #println("(L,l,l1)=($L,$l,$l1) \t s = $(SS[s_index]) \t w3j = $w3j")
                #println("res=$(res[s_index]) \t xis=$(VEC_FS[l+1][s_index]) \t qls=$(qls[s_index])\n")
            end
        end

        return ks, res
        =#

        for l in 0:length(XIMULT.multipoles)-1, l1 in 0:length(GENWIN.splines)-1

            w3j = convert(Float64, wigner3j(L, l, l1, 0, 0, 0))

            if (w3j ≈ 0.0) == false
                qls = [GENWIN.splines[l1+1](s) for s in SS]
                res += w3j^2 .* (2.0 * l1 + 1.0) .* VEC_FS[l+1] .* qls
                #s_index = findfirst(x->x>1000.0, SS)
                #println("(L,l,l1)=($L,$l,$l1) \t s = $(SS[s_index]) \t w3j = $w3j")
                #println("res=$(res[s_index]) \t xis=$(VEC_FS[l+1][s_index]) \t qls=$(qls[s_index])\n")
            end
        end

        _, pks = GaPSE.TwoFAST_PS_multipole(SS, res; L=L, cut_first_n=0, cut_last_n=0, kwargs...)

        return ks, pks

    elseif alg == :fftlog

        ks, _ = GaPSE.FFTLog_PS_multipole(SS, VEC_FS[1]; cut_first_n=0, cut_last_n=0, kwargs...)
        res = zeros(length(ks))

        #=
        for l in 0:length(XIMULT.multipoles)-1, l1 in 0:length(GENWIN.splines)-1

            w3j = convert(Float64, wigner3j(L, l, l1, 0, 0, 0))

            if (w3j ≈ 0.0) == false
                qls = [GENWIN.splines[l1+1](s) for s in SS]
                _, pks = GaPSE.FFTLog_PS_multipole(SS, VEC_FS[l+1] .* qls; L=L, cut_first_n=0, cut_last_n=0, kwargs...)
                res += w3j^2 .* (2.0 * l1 + 1.0) .* VEC_FS[l+1] .* qls
                #s_index = findfirst(x->x>1000.0, SS)
                #println("(L,l,l1)=($L,$l,$l1) \t s = $(SS[s_index]) \t w3j = $w3j")
                #println("res=$(res[s_index]) \t xis=$(VEC_FS[l+1][s_index]) \t qls=$(qls[s_index])\n")
            end
        end

        return ks, res
        =#

        for l in 0:length(XIMULT.multipoles)-1, l1 in 0:length(GENWIN.splines)-1

            w3j = convert(Float64, wigner3j(L, l, l1, 0, 0, 0))

            if (w3j ≈ 0.0) == false
                qls = [GENWIN.splines[l1+1](s) for s in SS]
                res += w3j^2 .* (2.0 * l1 + 1.0) .* VEC_FS[l+1] .* qls
                #s_index = findfirst(x->x>1000.0, SS)
                #println("(L,l,l1)=($L,$l,$l1) \t s = $(SS[s_index]) \t w3j = $w3j")
                #println("res=$(res[s_index]) \t xis=$(VEC_FS[l+1][s_index]) \t qls=$(qls[s_index])\n")
            end
        end

        _, pks = GaPSE.FFTLog_PS_multipole(SS, res; L=L, cut_first_n=0, cut_last_n=0, kwargs...)

        return ks, pks

    else
        throw(AssertionError(
            "The algorithm ':$alg' does not exist! The available ones are:\n" *
            "\t ':fftlog' (default), ':twofast' ."
        ))
    end
end




##########################################################################################92




"""
    print_PS_multipole_GenWin(
        ximult::Union{XiMultipoles,String}, genwin::Union{GenericWindow,String},
        out::String; alg::Symbol=:fftlog, L::Int=0, 
        cut_first_n::Int=0, cut_last_n::Int=0,
        kwargs...)

Compute and save in the file `out` 
the `L`-order Power Spectrum (PS) multipole for a generic window function, through the
following Fast Fourier Transform and the effective redshift approximation:

```math
    \\left\\langle \\hat{P}_L(k) \\right\\rangle = 
        \\frac{2 L + 1}{A} (-i)^L
        \\sum_{\\ell=0}^{\\infty} 
        \\sum_{\\ell_1=0}^{\\infty} 
        \\begin{pmatrix}
            L & \\ell & \\ell_1 \\\\
            0 & 0 & 0
        \\end{pmatrix}^2
        \\int_0^{\\infty}\\!\\mathrm{d} s \\, s^2 \\, \\xi_\\ell(s, s_{\\rm eff}) \\, 
        j_L(k s) \\, Q_{\\ell_1}(s) \\, ,
```

where:
- ``\\left\\langle \\hat{P}_L(k) \\right\\rangle`` is the order ``L`` Power Spectrum of the effect we 
  are interested in; we are basing this expression on the Yamamoto estimator (see Yamamoto (2000) and 
  Yamamoto (2006))
- ``A`` is a normalization constant
- the 2x3 matrix represents the Wigner-3j symbols
- ``\\xi_\\ell`` is the order ``\\ell`` multipole of the Two-Point Correlation Function (TPCF)
- ``j_L`` is the spherical Bessel function of order ``L``
- ``s_{\\mathrm{eff}}`` is the comoving distance associated with the effective redshift (see the 
  `TUTORIAL.ipynb` notebook)

``Q_{\\ell_1}`` can be easily estimated with FFT methods:

```math
\\begin{split}
    Q_{\\ell_1}(s) &:= \\int_0^{\\infty}\\mathrm{d}s_1 \\; s_1^2 \\;
    \\phi(s_1) \\; F_{\\ell_1}(s_1, s) \\\\
    F_{\\ell_1} (s_1 , s) &:= 
    \\int_{4\\pi} \\mathrm{d}\\Omega_{\\mathbf{\\hat{s}}} \\,  
    \\int_{4\\pi} \\mathrm{d}\\Omega_{\\mathbf{\\hat{s}}_1} \\,
    \\phi(s_2) \\, W(\\mathbf{\\hat{s}}_1) \\, W(\\mathbf{\\hat{s}}_2) \\,
    \\mathcal{L}_{\\ell_1}(\\mathbf{\\hat{s}} \\cdot \\mathbf{\\hat{s}}_1)  \\, .
\\end{split}
```

where:
- ``\\mathcal{L}_{\\ell_1}`` is the Legendre polynomial of order ``\\ell_1``
- ``\\mathrm{d}\\Omega_{\\mathbf{\\hat{s}}}`` is the infinitesimal solid angle pointing in the 
  direction of the versor ``\\mathbf{\\hat{s}}``
- ``\\phi(s)`` and  ``W(\\mathbf{\\hat{s}})`` are respectively the radial and angular part of your 
  window function (we remember that we assumed that such separability of the window function is possible)

Check Eq.(2.13) of Castorina, Di Dio for the theoretical explanation of this formula.
     
Currenlty, there are two algorithms you can choose in order to perform the computation; you can choose 
which one to use through the keyword value `alg`:
- `alg = :fftlog` (default and recommended option) will employ the [FFTLog](https://github.com/marcobonici/FFTLog.jl) 
  algorithm.
- `alg = :twofast` will employ the TwoFAST `xicalc` function of the [TwoFAST](https://github.com/hsgg/TwoFAST.jl) 
  Julia package. Note that in the computation the integration range ``0\\leq s \\leq \\infty`` 
  is reduced to `int_s_min ≤ s ≤ int_s_max`. This alogrithm is not the ideal choise, because TwoFAST is conceived
  for the direction PS -> TPCF, while is not 100% trustworthy for the other way round.

IMPORTANT: no matter which algorithm you choose, you will need to give the input data in a
LOGARITHMICALLY DISTRIBUTED scale. A linear distribution does not fit for the algorithms to apply.


## Inputs

- `ximult::Union{XiMultipoles,String}` : Two-Point Correlation functions to be used in the computation. You can
  provide either a `XiMultipoles` struct containing them or the String filename where they are stored (that will 
  be internally open with `XiMultipoles` too). 
  The file must have a structure analogous to the `genwin` one (see the next Inputs item). You can use 
  `create_file_for_XiMultipoles` to produce such a file.

- `genwin::Union{GenericWindow,String}` : multipoles ``Q_{\\ell_1}`` of the generic window function you want 
  to consider. You can provide either a `GenericWindow` struct containing them or the String filename 
  where they are stored (that will be internally open with `GenericWindow` too). 
  The file must have the following structure
  ```text
    \$ cat Ql_multipoles.txt 
    # Any comment line in the file must start with a #
    # you can have how many comment lines you want in the header; they 
    # will be all skipped.
    # Then you must provide in blank space separated columns:
    # - as first column, the comoving distance values, measured in Mpc/h_0
    # - from the second column onwards, all the Q_{\ell_1} multipoles you want;
    #   they must be ordered followinf the ascending multipole order (so \ell_1 = 0
    #   must be the 2 column), and you can go as further as you want in the multipole
    #   order
    # 
    # s [Mpc/h_0]      Q_{l1=0}      Q_{l1=1}      Q_{l1=2}      ...
    1.0     0.9999999999999999      1.445269850978461e-7      0.000011917268941324522    ...
    21.0    0.9832914433168294      -0.0025537781362117177  -0.0033199998619560947        ...
    41.0    0.9669175943095181      -0.004923364937797496      -0.006463561496567318        ...     
    ...     ...                  ...                      ...
  ```

- `out::String` : name of the file where the results must be stored.

##  Optional arguments

Depending on the algorithm you choose, the options would change. The options in common are:
- `pr::Bool=true` : want to print the automatic messages to the screen?
- `L::Int=0` : which multipole order should I use for this computation? IT MUST MATCH 
  THE MULTIPOLE ORDER OF THE INPUT TPCF!
- `cut_first_n::Int=0` and `cut_last_n::Int=0` : you can cout the first and/or last n elements
  of the input data, if they are highly irregular.

The specific ones for `alg = :fftlog` are:
- `ν::Union{Float64,Nothing} = nothing` : bias parameter, i.e. exponent used to "balance" the curve;
  if `nothing`, will be set automatically to `1.5`
- `n_extrap_low::Int = 500` and `n_extrap_high::Int = 500` : number of points to concatenate on the left/right
  of the input x-axis `ss` vector, logarithmically distributed with the same ratio of the left/right-edge
  elements of `ss`.
- `n_pad::Int = 500` : number of zeros to be concatenated both on the left and
  on the right of the input function. They stabilize a lot the algorithm.

The specific ones for `alg = :twofast` are:
- `epl::Bool=true` : do you want to extend the edges of the input vectors using two fitted
  power-laws (obtained from `EPLs`)
- `N_left::Int = 12` and `N_right::Int = 12` : number of points from left
  right edges to be used for the power law fitting in `EPLs`. They matters only
  if in the given input file ξ is not defined until the extremes of integration
  `int_s_min` and `int_s_max`.
- `int_s_min::Float64 = 1e-1` and `int_s_max::Float64 = 1e3`: extremes of integration; if `epl`
  is set to `false`, their values will be automatically set to `min(ss...)` and `max(ss...)`
  respectively. Their values do matter only if `epl=true`. 
- `p0_left=[-2.0, 1.0]` and `p0_right=[-2.0, 1.0]`: vectors with the initial values for the left/right 
  power-law fitting of `EPLs`; the power-law is in the form ``y = f(x) = b * x^s``, so the first vector 
  value is the initial value of ``s`` (and of course the second is the one of ``b``).
- `k0::Union{Nothing,Float64} = nothing` : starting point for the `xicalc` function; if `nothing`, 
  it will be set `k0 = 1.0 / max(ss...)`
- `right::Union{Float64,Nothing} = nothing` : do you want to cut the output elements with 
  `ks .> right`? if set to `nothing`, no cut will be done.
- `N::Int = 1024` : number of points to be used in Fourier transform 


See also: [`create_file_for_XiMultipoles`](@ref), [`XiMultipoles`](@ref), 
[`GenericWindow`](@ref),
[`V_survey`](@ref), [`A`](@ref), [`A_prime`](@ref),
[`EPLs`](@ref),  [`print_PS_multipole`](@ref), [`PS_multipole_GenWin`](@ref)
"""
function print_PS_multipole_GenWin(
    ximult::Union{XiMultipoles,String}, genwin::Union{GenericWindow,String},
    out::String; alg::Symbol=:fftlog, L::Int=0, pr::Bool=true,
    kwargs...)

    check_parent_directory(out)
    check_namefile(out)

    gwm = typeof(genwin) == String ? 
        "The Generic Window was taken from the file: $genwin" :
        "The Generic Window was taken as an object, not from a file."

    xmm = typeof(ximult) == String ?
        "The Xi Multipoles were taken from the file: $ximult" :
        "The Xi Multipoles were taken as an object, not from a file."

    pr && println("""\nI'm computing the PS_multipole_GenWin.\n""", xmm, "\n", gwm)

    time_1 = time()
    vec = PS_multipole_GenWin(ximult, genwin; L=L, pr=pr, alg=alg, kwargs...)
    time_2 = time()

    N = length(vec[1])

    isfile(out) && run(`rm $out`)
    open(out, "w") do io
        println(io, BRAND)

        println(io, "# Power Spectrum Multipole computation for a Generic Window function.")
        println(io, "# ", xmm)
        println(io, "# ", gwm)
        println(io, "#\n# For this PS_multipole_GenWin computation we set: ")
        println(io, "# \t multipole degree in consideration L = $L")
        println(io, "# \t algorithm chosen for the computation: :$alg")
        println(io, "# \t #points used in Fourier transform N = $N")
        println(io, "# computational time needed (in s) : $(@sprintf("%.4f", time_2-time_1))")
        print(io, "# kwards passed to \"print_PS_multipole_GenWin\": ")

        if isempty(kwargs)
            println(io, "none")
        else
            print(io, "\n")
            for key in keys(kwargs)
                println(io, "# \t\t$(key) = $(kwargs[key])")
            end
        end
        println(io, "# ")
        println(io, "# k [h_0/Mpc] \t \t  P [(Mpc/h_0)^3]")
        for (k, pk) in zip(vec[1], vec[2])
            println(io, "$k \t " * GaPSE.number_to_string(pk))
        end
    end
end

