using Pkg
using LinearAlgebra
using Plots
using Distributions
using Parameters
using CSV
using DataFrames, DataFramesMeta, DelimitedFiles

cd("/Users/bigfoot13770/Documents/UO ECON PROGRAM/ADRIFT/gentrification_sorting_model/gentrification_sorting_model_R")

# Read in crosswalks
tz_sea00 = CSV.read("tz_sea00.csv")
tz_det00 = CSV.read("tz_det00.csv")

# Read in tract data
seattle_tract = CSV.read("seattle_tract.csv")
detroit_tract = CSV.read("detroit_tract.csv")

# Blank df's to store results
seattle_pop = DataFrame(
    zip = unique(tz_sea00.ZIP),
    U00 = zeros(186),
    S00 = zeros(186)
)

detroit_pop = DataFrame(
    zip = unique(tz_det00.ZIP),
    U00 = zeros(240),
    S00 = zeros(240)
)

# Loop by zip and tract to match, calculate populations

for zip in tz_sea00.ZIP, tract in tz_sea00.TRACT
    pop = filter(y -> y[:GEOID] == tract, seattle_tract)
    ratio = filter(y -> y[:TRACT] == tract && y[:ZIP] == zip, tz_sea00)
    for z in 1:size(seattle_pop.zip)[1]
        if seattle_pop.zip[z] == ratio.ZIP[1] & size(ratio)[1] > 0
            seattle_pop.U00[z] += ratio.RES_RATIO[1] * (pop.POP100[1] - pop.P0570006[1])
            seattle_pop.S00[z] += ratio.RES_RATIO[1] * pop.P0570006[1]
        end
    end
end
