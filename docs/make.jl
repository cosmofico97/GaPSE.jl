push!(LOAD_PATH, "../src/")

using Documenter
using GaPSE

Documenter.makedocs(
     format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
     modules = [GaPSE],
     sitename = "GaPSE.jl",
     pages = [
          "Introduction" => "index.md",
          "The F window function" => "F_evaluation.md",
          "Background Data" => "BackgroundData.md",
          "Input Power Spectrum Tools" => "IPSTools.md",
          "Cosmology Struct" => "Cosmology.md",
          "Power Spectrum Multipoles" => "PowerSpectrum.md",
          "Auto Correlations" => "AutoCorrelations.md",
          "CrossCorrelations" => "CrossCorrelations.md",
          "Mathematical Utilities" => "MathUtils.md",
     ],
)

deploydocs(repo = "github.com/cosmofico97/GaPSE.jl.git")
