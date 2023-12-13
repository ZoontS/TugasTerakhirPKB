# Nama 1: Narendra Arkan Putra Darmawan
# NIM 1: 1313621043
# Nama 2: Muhammad Ramadhan Putra Pratama
# NIM 2: 1313621038

using DataFrames, Statistics, StatsBase, Serialization

function gini_impurity(labels)
    counts = values(countmap(labels))
    impurity = 1.0
    for count in counts
        prob = count / length(labels)
        impurity -= prob^2
    end
    return impurity
end

function find_best_split(data, labels)
    best_gini = 1.0
    best_feature = -1
    best_threshold = -1.0
    for feature_index in 1:size(data, 2)
        thresholds = sort(unique(data[:, feature_index]))
        for threshold in thresholds
            left_data, left_labels, right_data, right_labels = split_data(data, labels, feature_index, threshold)
            gini = (gini_impurity(left_labels) * size(left_data, 1) + gini_impurity(right_labels) * size(right_data, 1)) / size(data, 1)
            if gini < best_gini
                best_gini = gini
                best_feature = feature_index
                best_threshold = threshold
            end
        end
    end
    return best_feature, best_threshold
end

function build_tree(data, labels, depth=0, max_depth=5)
    if depth == max_depth || length(unique(labels)) == 1
        return mode(labels)
    end
    feature, threshold = find_best_split(data, labels)
    left_data, left_labels, right_data, right_labels = split_data(data, labels, feature, threshold)
    left = build_tree(left_data, left_labels, depth+1, max_depth)
    right = build_tree(right_data, right_labels, depth+1, max_depth)
    return feature, threshold, left, right
end

# Load data
data = deserialize(open("data_9m.mat", "r"))
data = convert(Array{Float64,2}, data)
labels = data[:, 5]
features = data[:, 1:4]

# Build the tree
tree = build_tree(features, labels)

