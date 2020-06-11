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

cps80 = CSV.read("cps80.csv")
cps80.naics = Int.(zeros(size(cps80)[1]))
for x in 1:size(cross90)[1], y in 1:size(cps80)[1]
    if cps80.IND90LY[y] == cross90.census90[x]
        cps80.naics[y] = cross90.naics[x]
    end
end

CSV.write("cps90.csv", cps90)
CSV.write("cps18.csv", cps18)
CSV.write("cps80.csv", cps80)

sic2naics = CSV.read("sic2naics.csv")
zbp94_sea = CSV.read("zbp94_sea.csv")
zbp94_det = CSV.read("zbp94_det.csv")

sic2naics.sic = div.(sic2naics.sic,10)

zbp94_sea.naics = Int.(zeros(size(zbp94_sea)[1]))

for x in 1:size(zbp94_sea)[1], y in 1:size(sic2naics)[1]
    if zbp94_sea.sic[x] == sic2naics.sic[y]
        zbp94_sea.naics[x] = sic2naics.naics[y]
    end
end

zbp94_det.naics = Int.(zeros(size(zbp94_det)[1]))

for x in 1:size(zbp94_det)[1], y in 1:size(sic2naics)[1]
    if zbp94_det.sic[x] == sic2naics.sic[y]
        zbp94_det.naics[x] = sic2naics.naics[y]
    end
end

CSV.write("zbp94_sea.csv", zbp94_sea)
CSV.write("zbp94_det.csv", zbp94_det)
