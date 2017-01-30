function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
    num_of_configurations = size(hidden_state, 2);
    visible_hidden_products = visible_state * hidden_state';
    visible_hidden_weight_products = rbm_w .* visible_hidden_products';
    
    G = sum(sum(visible_hidden_weight_products)) / num_of_configurations;
end

