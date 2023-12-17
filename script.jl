# Nama 1: Narendra Arkan Putra Darmawan
# NIM 1: 1313621043
# Nama 2: Muhammad Ramadhan Putra Pratama
# NIM 2: 1313621038

using DataFrames, Statistics, StatsBase, Serialization

function sumless_mean(data)
    averages = Matrix{Float16}(zeros, 3, 4)
    for col in axes(data, 2)
        n = 0
        mean = 0.0

        for x in data[:, col]
            n += 1
            delta = x - mean
            mean += delta / n
        end
        push!(averages, mean)
    end
    return averages
end

function euclidean_distance(data, averages)
    for col in axes(data, 2)
        for row in axes(data, 1)
            data[row, col] = abs(data[row, col] - averages[col])
        end
    end
    return data
end

data = deserialize(open("data_9m.mat", "r"))

averages = sumless_mean(data)

class1_ecd = euclidean_distance(class1_matrix, class1_averages)

