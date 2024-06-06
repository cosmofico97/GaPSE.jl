push!(LOAD_PATH, "../src/")

using Documenter
using GaPSE

Documenter.makedocs(
     format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
     modules=[GaPSE],
     sitename="GaPSE.jl",
     pages=[
          "Introduction" => "index.md",
          "The basic structure" => [
               "Background Data" => "BackgroundData.md",
               "Cosmology Parameters" => "CosmoParams.md",
               "Cosmology Struct" => "Cosmology.md",
          ],
          "The window function F and its integration" => [
               "Window F" => "WindowF.md",
               "Integrated Window F" => "WindowFIntegrated.md",
          ],
          "Calculating TPCFs multipoles" => [
               "GNC" => ["GNC_Correlations_1.md", "GNC_Correlations_2.md", "GNC_Correlations_3.md"],
               "LD" => ["LD_Correlations_1.md", "LD_Correlations_2.md"],
               "GNCxLD" => ["GNCxLD_Correlations_1.md", "GNCxLD_Correlations_2.md"],
               "LDxGNC" => ["LDxGNC_Correlations_1.md", "LDxGNC_Correlations_2.md"],
               "Kernels" => ["Kernels.md"]
          ],
          "Calculating TPCFs with the PP Approximation" => "PlaneParallelApprox.md",
          "Calculating Power Spectra" => "PowerSpectra.md",
          "Power Spectra for a generic window" => "PowerSpectraGenWin.md",
          "implication on PNG" => "PNG.md",
          "Utilities" => [
               "Dictionaries and names" => "Dicts.md",
               "Mathematical Utilities" => "MathUtils.md",
               "Cosmology Utilities" => "CosmoUtils.md",
               "Input Power Spectrum Tools" => "IPSTools.md",
               "Other Utilities" => "OtherUtils.md",
          ],
          "Spline" => [
               "MySpline" => "Spline.md",
               "Spline Theory" => "SplineTheory.md"
          ],
     ],
)

deploydocs(repo = "github.com/foglienimatteo/GaPSE.jl.git")
