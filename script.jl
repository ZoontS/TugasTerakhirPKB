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

function filter_correct_prediction(data, comparison)
    correct_size = 0
    incorrect_size = 0
    for row in axes(data, 1)
        if comparison[row] == true
            correct_size += 1
        else 
            incorrect_size += 1
        end
    end

    correct_data = Matrix{Float16}(undef, correct_size, 5)
    incorrect_data = Matrix{Float16}(undef, incorrect_size, 5)
    i = 1
    j = 1
    for row in axes(data, 1)
        if comparison[row] == true
            for col in 1:5
                correct_data[i, col] = data[row, col]
            end
            i += 1
        else
            for col in 1:5
                incorrect_data[j, col] = data[row, col]
            end
            j += 1
        end
    end

    return correct_data, incorrect_data
end

function correct_prediction_sumless_mean(data, col)
    averages = Matrix{Float16}(undef, 3, 1)

    for (i, val) in enumerate(1:3)
        filtered_data = data[data[:, 5] .== val, :]
        n = 0
        mean = 0.0
            
        for x in filtered_data[:, col]
            n += 1
            delta = x - mean
            mean += delta / n
        end 

        averages[i, 1] = mean
    end

    return averages
end

function calculate_correct_prediction_averages(data, comparison, accuracies)
    averages = Matrix{Float16}(undef, 3, 4)
    temp_accuracies = copy(accuracies)

    x1, remainder_data = filter_correct_prediction(data, comparison[:, argmax(temp_accuracies)])
    x1_averages = correct_prediction_sumless_mean(x1, argmax(temp_accuracies))
    averages[:, argmax(temp_accuracies)] = x1_averages
    temp_accuracies[argmax(temp_accuracies)] = 0.0

    x2, remainder_data = filter_correct_prediction(remainder_data, comparison[:, argmax(temp_accuracies)])
    x2_averages = correct_prediction_sumless_mean(x2, argmax(temp_accuracies))
    averages[:, argmax(temp_accuracies)] = x2_averages
    temp_accuracies[argmax(temp_accuracies)] = 0.0

    x3, remainder_data = filter_correct_prediction(remainder_data, comparison[:, argmax(temp_accuracies)])
    x3_averages = correct_prediction_sumless_mean(x3, argmax(temp_accuracies))
    averages[:, argmax(temp_accuracies)] = x3_averages
    temp_accuracies[argmax(temp_accuracies)] = 0.0

    x4, remainder_data = filter_correct_prediction(remainder_data, comparison[:, argmax(temp_accuracies)])
    x4_averages = correct_prediction_sumless_mean(x4, argmax(temp_accuracies))
    averages[:, argmax(temp_accuracies)] = x4_averages
    temp_accuracies[argmax(temp_accuracies)] = 0.0

    return averages
end


raw_data = deserialize(open("data_9m.mat", "r"))
println("Raw Data: ")
display(raw_data)

averages = calculate_sumless_mean(raw_data)
println("\nAverages: ")
display(averages)

predicted_classes = predict_class(raw_data[:, 1:4], averages)
println("\nPredicted Classes:")
display(predicted_classes)

comparison = compare_classes(predicted_classes, raw_data[:, 5])
println("\nComparison: ")
display(comparison)

accuracies = calculate_accuracy(comparison)
println("\nFeature 1 Accuracy: $(accuracies[1] * 100)%")
println("Feature 2 Accuracy: $(accuracies[2] * 100)%")
println("Feature 3 Accuracy: $(accuracies[3] * 100)%")
println("Feature 4 Accuracy: $(accuracies[4] * 100)%")

correct_prediction_averages = calculate_correct_prediction_averages(raw_data, comparison, accuracies)
println("\nNew Averages: ")
display(correct_prediction_averages)