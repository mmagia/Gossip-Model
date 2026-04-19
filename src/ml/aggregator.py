import torch
import copy

def aggregate(local_model, peer_weights, alpha=0.5):
    local_weights = local_model.state_dict()
    new_weights = copy.deepcopy(local_weights)
    
    with torch.no_grad():
        for name in local_weights.keys():
            if name in peer_weights:
                p_weight = peer_weights[name].to(local_weights[name].device)
                new_weights[name] = (1.0 - alpha) * local_weights[name] + alpha * p_weight
                
    return new_weights

def calculate_mse(weights_a, weights_b):
    total_mse = 0.0
    num_params = 0
    
    with torch.no_grad():
        for name in weights_a.keys():
            if name in weights_b:
                mse = torch.nn.functional.mse_loss(
                    weights_a[name].float(), 
                    weights_b[name].float()
                )
                total_mse += mse.item()
                num_params += 1
                
    return total_mse / num_params if num_params > 0 else 0.0