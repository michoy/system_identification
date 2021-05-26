using CSV
using DataFrames
using Plots
using StatsBase

print("hello World")

function load_df(file_path::AbstractString) 
    return DataFrame(CSV.File(file_path))
end

df = load_df("data/preprocessed/yaw-1.csv")

print(df)
