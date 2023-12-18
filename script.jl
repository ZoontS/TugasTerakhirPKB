# Nama 1: Narendra Arkan Putra Darmawan
# NIM 1: 1313621043
# Nama 2: Muhammad Ramadhan Putra Pratama
# NIM 2: 1313621038

using DataFrames, Statistics, Serialization

function calculate_sumless_mean(data)
    averages = Matrix{Float16}(undef, 3, 4)

    for (i, val) in enumerate(1:3)
        filtered_data = data[data[:, 5] .== val, :]
        
        for col in 1:4
            n = 0
            mean = 0.0
            
            for x in filtered_data[:, col]
                n += 1
                delta = x - mean
                mean += delta / n
            end 

            averages[i, col] = mean
        end
    end
    return averages
end

function predict_class(data, averages)
    predicted_classes = Matrix{Float16}(undef, size(data, 1), 4)

    for row in axes(data, 1)

        for col in axes(data, 2)
            distances = zeros(Float16, 3)

            for i in 1:3
                distances[i] = abs(data[row, col] - averages[i, col])
            end
            
            predicted_classes[row, col] = argmin(distances)
        end
    end

    return predicted_classes
end

function compare_classes(predicted_classes, actual_classes)
    comparison = Matrix{Bool}(undef, size(predicted_classes, 1), 4)

    for row in axes(predicted_classes, 1)
        for col in axes(predicted_classes, 2)
            comparison[row, col] = predicted_classes[row, col] == actual_classes[row]
        end
    end

    return comparison
end

function calculate_accuracy(comparison)
    accuracy = zeros(Float32, 4)

    for i in 1:4
        accuracy[i] = mean(comparison[:, i])
    end

    return accuracy
end

raw_data = deserialize(open("data_9m.mat", "r"))

averages = calculate_sumless_mean(raw_data)

predicted_classes = predict_class(raw_data[:, 1:4], averages)

comparison = compare_classes(predicted_classes, raw_data[:, 5])

accuracies = calculate_accuracy(comparison)

println("Raw Data: ")
display(raw_data)

println("\nAverages: ")
display(averages)

println("\nPredicted Classes:")
display(predicted_classes)

println("\nComparison: ")
display(comparison)

println("\nFeature 1 Accuracy: $(accuracies[1] * 100)%")
println("Feature 2 Accuracy: $(accuracies[2] * 100)%")
println("Feature 3 Accuracy: $(accuracies[3] * 100)%")
println("Feature 4 Accuracy: $(accuracies[4] * 100)%")