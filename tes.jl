using DataFrames, Statistics, StatsBase, Serialization

raw_data = deserialize(open("data_9m.mat", "r"))
data = DataFrame(raw_data, [:x1, :x2, :x3, :x4, :y])

class1 = data[data[!, :y] .== 1.0, :]

class1_matrix = Matrix{Float16}(class1[:, 1:4])

display(axes(class1_matrix, 2))