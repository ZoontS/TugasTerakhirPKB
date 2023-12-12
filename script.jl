using DataFrames, Statistics, Serialization

raw_data = deserialize(open("data_9m.mat", "r"))
data = DataFrame(raw_data, :auto)

