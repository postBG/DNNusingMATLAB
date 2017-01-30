function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    v0 = visible_data;
    
    probability_for_h0 = visible_state_to_hidden_probabilities(rbm_w, v0);
    h0 = sample_bernoulli(probability_for_h0);
    
    probability_for_v0 = hidden_state_to_visible_probabilities(rbm_w, h0);
    v1 = sample_bernoulli(probability_for_v0);
    
    probability_for_h1 = visible_state_to_hidden_probabilities(rbm_w, v1);
    
    
    data_goodness_gradient = configuration_goodness_gradient(v0, h0);
    sample_goodness_gradient = configuration_goodness_gradient(v1, probability_for_h1);
    
    ret = data_goodness_gradient - sample_goodness_gradient;
end
