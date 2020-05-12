using Pkg
using LinearAlgebra
using Plots
using Distributions
using Parameters
using CSV
using DataFrames, DataFramesMeta, DelimitedFiles

cd("/Users/bigfoot13770/Documents/UO ECON PROGRAM/ADRIFT/gentrification_sorting_model/gentrification_sorting_model_R")

cross90 = CSV.read("crosswalk90.csv")
cps90 = CSV.read("cps90.csv")

cps90.naics = Int.(zeros(size(cps90)[1]))

for x in 1:size(cross90)[1], y in 1:size(cps90)[1]
    if cps90.IND[y] == cross90.census90[x]
        cps90.naics[y] = cross90.naics[x]
    end
end

cross18 = CSV.read("crosswalk18.csv")
cps18 = CSV.read("cps18.csv")

cps18.naics = Int.(zeros(size(cps18)[1]))

for x in 1:size(cross18)[1], y in 1:size(cps18)[1]
    if cps18.IND[y] == cross18.census18[x]
        cps18.naics[y] = cross18.naics[x]
    end
end

CSV.write("cps90.csv", cps90)
CSV.write("cps18.csv", cps18)
