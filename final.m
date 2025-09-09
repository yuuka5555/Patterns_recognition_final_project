function accuracy = final()

    people = 40;
    images_per_person = 10; 
    pca_components = 100;
    hidden_neurons = 150;

    [Row_FACE_Data_train, Row_FACE_Data_test] = load_data(people, images_per_person);
    
    %PCA
    MofD = mean(Row_FACE_Data_train, 1);
    NorData_train = Row_FACE_Data_train - MofD; 
    CMatrix = NorData_train' * NorData_train / size(NorData_train, 1);
    [vec, val]=eig(CMatrix);
    [~, sorted_indices] = sort(diag(val), 'descend');
    PCA_components = vec(:, sorted_indices(1:pca_components));
    
    %project
    PCA_result_train = NorData_train * PCA_components;
    PCA_result_test = (Row_FACE_Data_test - MofD) * PCA_components;

    % Min-Max
    min_data = min(PCA_result_train, [], 1)
    max_data = max(PCA_result_test, [], 1)
    TrainNorm = (PCA_result_train - min_data) ./ (max_data - min_data + 1e-9);
    TestNorm = (PCA_result_test - min_data) ./ (max_data - min_data + 1e-9);

    % Prepare labels
    labels_train = repelem(1:people, images_per_person/2)';
    labels_test = repelem(1:people, images_per_person/2)';
    num_classes = people;

    Y_train = one_hot_encode(labels_train, num_classes);
    Y_test = one_hot_encode(labels_test, num_classes);

    % weight
    w1 = randn(pca_components, hidden_neurons) * sqrt(2.0 / pca_components); % Input to Hidden
    b1 = zeros(1, hidden_neurons);
    w2 = randn(hidden_neurons, num_classes) * sqrt(2.0 / hidden_neurons); % Hidden to Output
    b2 = zeros(1, num_classes);
    
    % Training parameters
    learning_rate = 0.01;  
    num_epochs = 300;;       

    for epoch = 1:num_epochs
        sum_hid =  TrainNorm * w1 + b1;
        Ahid = sigmoid(sum_hid);

        sum_out = Ahid * w2 + b2;
        Aout = softmax_rows(sum_out);

        % Compute cross-entropy loss
        epsilon = 1e-12;
        loss = -sum(sum(Y_train .* log(Aout + epsilon))) / size(Aout, 1);

        % Backpropagation
        % Output layer gradients
        delta_Aout = Aout - Y_train;
        dw2 = Ahid' * delta_Aout;
        db2 = sum(delta_Aout, 1);

        % Hidden layer gradients
        d_trans = delta_Aout * w2';
        delta_Ahid = d_trans .* sigmoid_derivative(sum_hid);
        dw1 = TrainNorm' * delta_Ahid;
        db1 = sum(delta_Ahid, 1);

        % Update parameters
        w2 = w2 - learning_rate * dw2;
        b2 = b2 - learning_rate * db2;
        w1 = w1 - learning_rate * dw1;
        b1 = b1 - learning_rate * db1;
 
        if mod(epoch, 10) == 0
            fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
        end
    end
 
    % Evaluate on test set
    Z1_test = TestNorm * w1 + b1;
    A1_test = sigmoid(Z1_test);
    Z2_test = A1_test * w2 + b2;
    Y_hat_test = softmax_rows(Z2_test);
    [~, preds] = max(Y_hat_test, [], 2);
    correct = sum(preds == labels_test);
    Accuracy = (correct / length(labels_test)) * 100;
    fprintf('Recognition Accuracy: %.2f%%\n', Accuracy);

end

function [Row_FACE_Data_train, Row_FACE_Data_test] = load_data(people, images_per_person)
    Row_FACE_Data_train = [];
    Row_FACE_Data_test = [];
 
    for k = 1:people
        for m = 1:images_per_person
            PathString = ['D:' '\' 'matlab' '\' 'PCA' '\' 'ORL3232' '\' num2str(k) '\' num2str(m) '.bmp'];
 
            if isfile(PathString)
                ImageData = imread(PathString);
                ImageData = double(ImageData);
                RowConcatenate = reshape(ImageData', 1, []);
 
                % 1,3,5,7,9 for training, others for testing
                if mod(m, 2) == 1
                    Row_FACE_Data_train = [Row_FACE_Data_train; RowConcatenate];
                else
                    Row_FACE_Data_test = [Row_FACE_Data_test; RowConcatenate];
                end
            else
                warning(['File not found: ' PathString]);
            end
        end
    end
end

function o_h = one_hot_encode(labels, num_classes)
    N = length(labels);
    o_h = zeros(N, num_classes);
    for i = 1:N
        o_h(i, labels(i)) = 1;
    end
end

function Y = softmax_rows(Z)
    maxZ = max(Z, [], 2);
    Z = Z - repmat(maxZ, 1, size(Z, 2));
    expZ = exp(Z);
    Y = expZ ./ repmat(sum(expZ, 2), 1, size(expZ, 2));
end

function Y = sigmoid(X)
    Y = 1 ./ (1 + exp(-X));
end

function Y = sigmoid_derivative(X)
    s = sigmoid(X);
    Y = s .* (1 - s);
end