# using softplus in the optimisation for umerical stability
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Calibrator():
  

  #Add flags here as instance attributes
  def __init__(self, deep_ensemble, adjusting_alpha_method = "convex_comb", clamping_alphas = False, logits_based_adjustments = True):

    valid_methods = [ "convex_comb", "pure_logits", "convex_comb_no_exp", "convex_comb_global"]
    if adjusting_alpha_method.lower() not in valid_methods:
      raise ValueError(f"Invalid adjusting_alpha_method: {adjusting_alpha_method}. Valid methods are {valid_methods}")

    self.ensemble = deep_ensemble
    self.device = self.ensemble.device
    self.aa_method = adjusting_alpha_method.lower()
    self.clamping_alphas = clamping_alphas
    self.logits_based_adjustments = logits_based_adjustments


  ##################################
  # Helper functions BEGIN
  ##################################

  #---------------------------------
  # Moment matching Dirichlet (Graph)
  #---------------------------------

  def fit_dirichlet_moment_matching(self, data):
    """
    Fit a Dirichlet distribution using moment matching.

    Args:
        data: shape (num_models, batch_size, num_classes) - probabilities (must sum to 1 along -1)

    Returns:
        alphas: shape (batch_size, num_classes) - estimated Dirichlet parameters
    """
    # Add a small value to probabilities to avoid -inf during log computation
    data = torch.add(data, 1e-6)
    data = data / data.sum(dim=-1, keepdim=True)  # Normalize probabilities

    # Compute first moment (mean of probabilities)
    mean_probs = data.mean(dim=0)  # Shape: (batch_size, num_classes)

    # Compute variance
    var_probs = data.var(dim=0)  # Shape: (batch_size, num_classes)

    # Compute alpha0 (sum of all alphas) using method of moments
    mean_sum = mean_probs.sum(dim=-1, keepdim=True)  # Shape: (batch_size, 1)
    numerator = mean_sum * (1 - mean_probs).sum(dim=-1, keepdim=True)  # Shape: (batch_size, 1)
    denominator = var_probs.sum(dim=-1, keepdim=True)  # Shape: (batch_size, 1)
    alpha0 = numerator / (denominator + 1e-6) - 1  # Shape: (batch_size, 1)

    # Compute individual alphas
    alphas = mean_probs * alpha0  # Shape: (batch_size, num_classes)

    return alphas

  #---------------------------------
  # Scaling/clamping alphas
  #---------------------------------

  def scaling_alphas(self, alphas):
    """
    Input and Output:

      Torch alphas: size [batch_size, num_classes]

    """

    assert torch.all(alphas > 0), "All alpha values must be positive."

    if self.clamping_alphas:
      alphas_sorted, _ = torch.sort(alphas, dim=-1) #returns idx
      alphas_second_min = alphas_sorted[:, 1:2]

      mask = alphas_second_min < 1
      scalar = (1/alphas_second_min).masked_fill(~mask, 1.0) #Complement gets 1.0  shape [batch_size]
      alphas_scaled = alphas * scalar

      return alphas_scaled

    #if false do nothing
    else:
      return alphas

  #---------------------------------
  # Average standard deviation of logits - used for alpha adjustments
  #---------------------------------

  def average_std_logits(self, model_outputs):
    """
    Args:
        logits: shape [num_models, batch_size, num_classes]
    Returns:
        mean_std: shape [batch_size]
    """

    std_logits = torch.var(model_outputs, dim=0)  # Shape: [batch_size, num_classes]
    mean_std = torch.sqrt(torch.mean(std_logits, dim=-1))  # Shape: [batch_size]
    return mean_std


  #---------------------------------
  # Adjust alphas
  #---------------------------------

  def adjust_alphas(self, alphas, epistemic_uncertainty, c2, epi_scalar):
    """
    Dispatch to one of our alpha-adjustment methods based on `method`.
    NOTE, the epistemic uncertainty can be exchanged with other metrics of uncertainty

    method:
        ["convex_comb",
        "pure_logits",
        "convex_comb_no_exp",
        "convex_comb_global"]
    """

    if self.aa_method == "convex_comb":
        return self.adjust_alphas_CC(alphas, epistemic_uncertainty, c2, epi_scalar)
    elif self.aa_method == "pure_logits":
        return None
    elif self.aa_method == "convex_comb_no_exp":
       return self.adjust_alphas_CC_without_exp(alphas, epistemic_uncertainty, c2, epi_scalar)
    elif self.aa_method == "convex_comb_global":
      return self.adjusting_alphas_CC_global(alphas, epistemic_uncertainty, c2, epi_scalar)
    else:
        raise ValueError(f"Unknown method: {self.aa_method}. Valid options are ['convex_comb','pure_logits', 'convex_comb_no_exp', 'convex_comb_global']")


  #Convex combination
  def adjust_alphas_CC(self, alphas, epistemic_uncertainty, c2, scalar):

    assert torch.all(alphas >0), "All alpha values must be positive"

    temp_c2 = torch.exp(c2 * epistemic_uncertainty* scalar)  # [batch_size]
    rho = (1.0 / temp_c2).unsqueeze(1)  # [batch_size, 1]
    adjusted_alphas = (1 - rho) * 1 + rho * alphas  # Broadcasting automatically
    return adjusted_alphas

  #Convex combination without exponentiating epistemic uncertainty
  def adjust_alphas_CC_without_exp(self, alphas, epistemic_uncertainty, c2, scalar):

    assert torch.all(alphas >0), "All alpha values must be positive"

    #temp_c2 follows teh desiderata
    temp_c2 = (c2 * epistemic_uncertainty* scalar)  #[batch_size]
    rho = (1.0 / (1 + temp_c2)).unsqueeze(1)  # [batch_size, 1]
    adjusted_alphas = (1 - rho) * 1 + rho * alphas  # Broadcasting automatically
    return adjusted_alphas

  def adjusting_alphas_CC_global(self, alphas, epistemic_uncertainty, c2, scalar):

    assert torch.all(alphas >0), "All alpha values must be positive"

    adjusted_alphas = c2 * 1 + (1- c2) * alphas  # Broadcasting automatically
    return adjusted_alphas


  #pure_logits
  def adjust_logits(self, logits, c2):
    """
    Input needs to be of shape [num_models, batch_size, num_classes]
    """
    mean_logits = logits.mean(dim=0)  # Shape: [batch_size, num_classes]
    updated_logits = mean_logits + c2* (logits - mean_logits)  # Shape: [num_models, batch_size, num_classes]
    return updated_logits


  #---------------------------------
  # Uncertainty estimation - Based only on Dirichlet
  #---------------------------------

  def mi_dir_monte_carlo(self, alphas, num_samples= 1e4):
    """
    Monte Carlo approximation of the Dirichlet MI
      Args: alphas [batch_size, num_classes]
      Returns: aleatoric, epistemic, total [batch_size]
    """

    num_samples = int(num_samples)

    dirichlet = torch.distributions.Dirichlet(alphas) # Shape: [batch_size, num_classes]
    prob_samples = dirichlet.sample([num_samples])  # Shape: [num_samples, batch_size, num_classes]

    # MC approximation of entropy
    aleatoric = torch.mean(-torch.sum(prob_samples * torch.log(prob_samples + 1e-8), dim=-1), dim=0)  # Shape: [batch_size]
    aleatoric_median = torch.median(-torch.sum(prob_samples * torch.log(prob_samples + 1e-8), dim=-1), dim=0).values
    total = -torch.sum(torch.mean(prob_samples, dim=0) * torch.log(torch.mean(prob_samples, dim=0) + 1e-8), dim=-1)  # Shape: [batch_size]
    epistemic = total - aleatoric
    return aleatoric, epistemic, total, aleatoric_median

  #---------------------------------
  # Uncertainty estimation - partly with Deep Ensemble - used in NLL
  #---------------------------------

  def mi_dir_de_mix(self, member_probs, mean_probs):
    """
    member_probs: [num_models, batch_size, num_classes]
    mean_probs : [batch_size, num_classes]
    returns: (all uncertainties) each [batch_size]
    """

    model_entropies = -torch.sum(member_probs * torch.log(member_probs + 1e-8), dim=-1) # [num_models, batch_size]
    aleatoric = model_entropies.mean(dim=0) # [batch_size]
    total = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1) # [batch_size]
    epistemic = total - aleatoric
    return aleatoric, epistemic, total



  ##################################
  # Helper functions END
  ##################################


  #---------------------------------
  # Compute NLL (Fully in Torch)
  #---------------------------------
  def validation_nll(self, val_loader, c1, c2, epi_scalar):
    """
    c1: float for temperature scaling
    c2: torch.Tensor with requires_grad
    returns: a torch scalar (the average NLL over the entire validation set)
    """
    total_loss = 0.0
    total_count = 0

    self.ensemble.eval()

    for inputs, labels in val_loader:
      inputs, labels = inputs.to(self.device), labels.to(self.device)

      if self.aa_method == "convex_comb" or self.aa_method == "convex_comb_no_exp" or self.aa_method == "convex_comb_global":
        with torch.no_grad():
          model_outputs = []
          for model in self.ensemble.models:
            outputs = model(inputs)
            scaled_outputs = outputs / c1  # temperature scaling
            model_outputs.append(scaled_outputs)

          model_outputs = torch.stack(model_outputs)  # [num_models, batch_size, num_classes]
          member_probs = F.softmax(model_outputs, dim=-1)
          mean_probs = member_probs.mean(dim=0)  # [batch_size, num_classes]

          alphas = self.fit_dirichlet_moment_matching(member_probs)
          alphas = self.scaling_alphas(alphas)

        alphas = alphas.to(self.device)

        if not self.logits_based_adjustments:
          _, epistemic_uncertainty, _ = self.mi_dir_de_mix(member_probs, mean_probs)
          adjusted_alphas = self.adjust_alphas(alphas, epistemic_uncertainty, c2, epi_scalar)
        else:
          average_std = self.average_std_logits(model_outputs)
          adjusted_alphas = self.adjust_alphas(alphas, average_std, c2, epi_scalar)

        adjusted_mean_probs = adjusted_alphas / adjusted_alphas.sum(dim=1, keepdim=True)
        log_probs = torch.log(adjusted_mean_probs + 1e-12)
        nll_batch = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

      elif self.aa_method == "pure_logits":
        model_outputs = []
        for model in self.ensemble.models:
          outputs = model(inputs)
          scaled_outputs = outputs / c1
          model_outputs.append(scaled_outputs)
        model_outputs = torch.stack(model_outputs)
        adjusted_logits = self.adjust_logits(model_outputs, c2)
        member_probs = F.softmax(adjusted_logits, dim=-1)
        mean_probs = member_probs.mean(dim=0)
        log_probs = torch.log(mean_probs + 1e-12)
        nll_batch = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

      else:
        raise ValueError(f"Unknown aa_method: {self.aa_method}")

      total_loss += nll_batch.sum()
      total_count += inputs.size(0)

    # Gather in outer for loop
    final_nll = total_loss / total_count
    return final_nll  # Return after processing all batches
  
  #---------------------------------
  # Compute NLL efficiently for grid search
  #---------------------------------

  def precompute_val_outputs(self, val_loader):
    """
    Performs one full forward pass over the validation set and outputs member logits and labels.
    
    Returns:
        model_outputs: Tensor of shape [num_models, total_samples, num_classes]
        labels: Tensor of shape [total_samples]
    """
    self.ensemble.eval()
    outputs_list = []
    labels_list = []
    with torch.no_grad():
       for inputs, labels in val_loader:
          inputs, labels = inputs.to(self.device), labels.to(self.device)
          batch_outputs = []
          for model in self.ensemble.models:
            outputs = model(inputs)
            batch_outputs.append(outputs)
          batch_outputs = torch.stack(batch_outputs)    # [num_models, batch_size, num_classes]
          outputs_list.append(batch_outputs)
          labels_list.append(labels)
    # Concatenate all batches along the batch dimension:
    model_outputs = torch.cat(outputs_list, dim=1)  # [num_models, total_samples, num_classes]
    labels = torch.cat(labels_list, dim=0)           # [total_samples]
    return model_outputs, labels


  def validation_nll_grid_from_precomputed(self, model_outputs, labels, c1, c2, epi_scalar):
    """ 
    Returns:
        final_nll: A torch scalar representing the average NLL over the validation set.
    """
    if self.aa_method in ["convex_comb", "convex_comb_no_exp", "convex_comb_global"]:
      # Apply temperature scaling
      scaled_outputs = model_outputs / c1               # [num_models, total_samples, num_classes] 
      member_probs = F.softmax(scaled_outputs, dim=-1)  # [num_models, total_samples, num_classes]
      mean_probs = member_probs.mean(dim=0)             # [total_samples, num_classes]

      alphas = self.fit_dirichlet_moment_matching(member_probs) # [total_samples, num_classes]  
      alphas = self.scaling_alphas(alphas)                      # [total_samples, num_classes]
      # Adjust alphas based on uncertainty
      if not self.logits_based_adjustments:
        _, epistemic_uncertainty, _ = self.mi_dir_de_mix(member_probs, mean_probs)
        adjusted_alphas = self.adjust_alphas(alphas, epistemic_uncertainty, c2, epi_scalar)
      else:
        average_std = self.average_std_logits(scaled_outputs)
        adjusted_alphas = self.adjust_alphas(alphas, average_std, c2, epi_scalar)

      adjusted_mean_probs = adjusted_alphas / adjusted_alphas.sum(dim=1, keepdim=True)  # [total_samples, num_classes]
      log_probs = torch.log(adjusted_mean_probs + 1e-12)
      nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    
    elif self.aa_method == "pure_logits":
      scaled_outputs = model_outputs / c1
      adjusted_logits = self.adjust_logits(scaled_outputs, c2)
      member_probs = F.softmax(adjusted_logits, dim=-1)
      mean_probs = member_probs.mean(dim=0)
      log_probs = torch.log(mean_probs + 1e-12)
      nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    else:
      raise ValueError(f"Unknown aa_method: {self.aa_method}")
    
    final_nll = nll.mean()
    return final_nll

#---------------------------------
# GRID SEARCH
#---------------------------------

# Efficient grid search for c1, c2, and epi_scalar

  def grid_search_c1_c2_precomputed(self, val_loader, c1_vals, c2_vals, epi_scalar_vals):
    """
    Performs grid search over c1, c2, and epi_scalar using precomputed model outputs.
    """
    results = []
    best_nll = float('inf')
    best_params = {"c1": None, "c2": None, "epi_scalar": None, "nll_val": None}

    model_outputs, labels = self.precompute_val_outputs(val_loader)
    
    for c1 in c1_vals:
      for c2 in c2_vals:
        for epi_scalar in epi_scalar_vals:
          final_nll = self.validation_nll_grid_from_precomputed(model_outputs, labels, c1, c2, epi_scalar)
          results.append((c1, c2, epi_scalar, final_nll.item()))
          if final_nll.item() < best_nll:
            best_nll = final_nll.item()
            best_params = {"c1": c1, "c2": c2, "epi_scalar": epi_scalar, "nll_val": best_nll}
    
    return results, best_params
  
  
  # Less efficient grid search for c1, c2, and epi_scalar
  
  def grid_search_c1_c2(self, val_loader, c1_vals, c2_vals, epi_scalar_vals):
    """
    Full grid search over c1, c2, and epi_scalar values to find the best combination.
    """

    #NOTE when running this grid search c2 should start at 1.0 for the convex_comb_no_unc method
    # Else 0.0 suffices

    results = []
    best_nll = float('inf')
    best_params = {"c1": None, "c2": None, "epi_scalar": None, "nll": None}

    for c1 in c1_vals:
      for c2 in c2_vals:
        for epi_scalar in epi_scalar_vals:
          # Compute the NLL for the current combination
          final_nll = self.validation_nll(val_loader, c1, c2, epi_scalar)
          results.append((c1, c2, epi_scalar, final_nll.item()))

          # Update the best parameters if this combination is better
          if final_nll.item() < best_nll:
            best_nll = final_nll.item()
            best_params = {"c1": c1, "c2": c2, "epi_scalar": epi_scalar, "nll_val": best_nll}

    return results, best_params




  #---------------------------------
  # OPTIMIZING
  #---------------------------------
  # inefficient as we use grid for c1, but could optimize both c1 and c2

  def optimize_c1_c2(self, val_loader, c1_vals, epi_scalar_lr_tuples, c2_init, c2_steps=50, tolerance = 1e-6, patience = 2):
    """
        Uses softplus for numerical stability
        Initialisation, might have to be adapted - 0.0 for convex_comb and 1.0 for rest as default
    """

    results = []
    best_nll = float('inf')

    theta_init = np.log(np.exp(c2_init) - 1)

    for c1 in c1_vals:
      for epi_scalar, lr in epi_scalar_lr_tuples:
        # NOTE will try to optimize without getting negative numbers
        theta = torch.tensor(theta_init, device=self.device, requires_grad=True)
        #c2 = torch.tensor(c2_init, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([theta], lr=lr) #requires iterator

        no_improvement = 0
        prev_nll = float('inf')

        #prev_nll = float("inf")
        for step in range(c2_steps):
          optimizer.zero_grad()
          # need to undo the transformation
          c2 = F.softplus(theta)
          nll_tensor = self.validation_nll(val_loader, c1, c2, epi_scalar)
          nll_val = nll_tensor.item()

          # Check if improvement is below the tolerance
          if abs(prev_nll - nll_val) < tolerance:
            no_improvement += 1
          else:
            no_improvement = 0  # reset

          prev_nll = nll_val

          # break if it does not improve
          if no_improvement >= patience:
            break

          # if not we continue
          nll_tensor.backward()
          optimizer.step()

        # Evaluate final
        c2 = F.softplus(theta)
        final_nll = self.validation_nll(val_loader, c1, c2, epi_scalar)
        results.append((c1, c2.item(), epi_scalar, lr, final_nll.item())) # not requiring grad

        if final_nll.item() < best_nll:
          best_nll = final_nll.item()
          best_params = {"c1": c1, "c2": c2.item(), "epi_scalar": epi_scalar, "lr": lr, "nll_val": best_nll}

    return results, best_params


  

  #---------------------------------
  # PREDICTING
  #---------------------------------
   
  def predict(self, loader, c1_prim, c2_prim, epi_scalar_prim):
    """
        Compute predictions and uncertainty measures for the given data.

        Args:
        loader (DataLoader): Data loader for inference.
        c1_prim (float): Temperature scaling factor.
        c2_prim (float): Scaling factor for alpha adjustments.
        epi_scalar_prim (float): Scalar applied to uncertainty measures.

        Returns:
        dict: Dictionary containing:
            - "alphas": Dirichlet parameters (shape: [num_samples, num_classes])
            - "member_probs": Ensemble member probabilities (shape: [num_models, num_samples, num_classes])
            - "ensemble_probs": Full predictive probability distribution (shape: [num_samples, num_classes])
            - "probabilities": Maximum predictive probability per sample (shape: [num_samples])
            - "member_logits": Model logits (shape: [num_models, num_samples, num_classes])
            - "predictions": Predicted class indices (shape: [num_samples])
            - Uncertainty measures: "epistemic_uncertainty", "aleatoric_uncertainty",
              "aleatoric_median", "total_uncertainty" (each with shape: [num_samples])
            - "nll": Negative log-likelihood values (shape: [num_samples], if labels are available)
    """
    alphas_list = []
    member_probs_list = []
    ensemble_probs_list = []   # Full predictive probability distributions per batch
    max_probs_list = []        # Maximum probability per sample (scalar per sample)
    predictions_list = []
    member_logits_list = []
    nll_list = []

    epistemic_uncertainty_list = []
    aleatoric_uncertainty_list = []
    aleatoric_median_list = []
    total_uncertainty_list = []

    self.ensemble.eval()  # Set ensemble to evaluation mode
    with torch.no_grad():
        for batch in loader:
            # Extract inputs and labels (if available)
            if len(batch) == 1:
                inputs = batch[0].to(self.device)
                labels = None
            else:
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

            # Collect scaled model logits for this batch
            batch_model_outputs = []
            for model in self.ensemble.models:
                outputs = model(inputs)
                scaled_outputs = outputs / c1_prim
                batch_model_outputs.append(scaled_outputs)
            batch_model_outputs = torch.stack(batch_model_outputs)  # [num_models, batch_size, num_classes]

            if self.aa_method == "pure_logits":
                adjusted_logits = self.adjust_logits(batch_model_outputs, c2_prim)
                batch_member_probs = F.softmax(adjusted_logits, dim=-1)
                batch_ensemble_probs = batch_member_probs.mean(dim=0)  # [batch_size, num_classes]
                
                # Fit Dirichlet distribution on member probabilities
                batch_alphas = self.fit_dirichlet_moment_matching(batch_member_probs)
                batch_adjusted_alphas = self.scaling_alphas(batch_alphas)
            else:
                batch_member_probs = F.softmax(batch_model_outputs, dim=-1)  # [num_models, batch_size, num_classes]
                batch_mean_probs = batch_member_probs.mean(dim=0)  # [batch_size, num_classes]
                
                # Fit Dirichlet distribution
                batch_alphas = self.fit_dirichlet_moment_matching(batch_member_probs)
                batch_adjusted_alphas = self.scaling_alphas(batch_alphas)
                
                # Adjust alphas based on uncertainty measure
                if not self.logits_based_adjustments:
                    _, batch_epistemic_uncertainty, _ = self.mi_dir_de_mix(batch_member_probs, batch_mean_probs)
                    batch_adjusted_alphas = self.adjust_alphas(batch_adjusted_alphas, 
                                                                batch_epistemic_uncertainty, c2_prim, epi_scalar_prim)
                else:
                    batch_avg_std = self.average_std_logits(batch_model_outputs)
                    batch_adjusted_alphas = self.adjust_alphas(batch_adjusted_alphas, 
                                                                batch_avg_std, c2_prim, epi_scalar_prim)
                
                batch_ensemble_probs = batch_adjusted_alphas / batch_adjusted_alphas.sum(dim=1, keepdim=True)
            
            # NOTE computing ofr all adjustments here - might be unnecessary for "pure_logits"
            batch_max_probs, batch_predictions = torch.max(batch_ensemble_probs, dim=1)
            
            # Compute uncertainty metrics via Monte Carlo sampling from the Dirichlet
            batch_aleatoric_uncertainty, batch_epistemic_uncertainty, batch_total_uncertainty, batch_aleatoric_median = self.mi_dir_monte_carlo(batch_adjusted_alphas)
            
            # If labels are available, compute NLL for this batch
            if labels is not None:
                batch_nll = -torch.gather(torch.log(batch_ensemble_probs + 1e-12), dim=1, 
                                          index=labels.unsqueeze(1)).squeeze(1)
                nll_list.append(batch_nll.cpu())
                #other options with same result
                #nll_batch = -torch.log(predictive_mean_probs[torch.arange(labels.shape[0]), labels] + 1e-12)
                #nll_batch = F.nll_loss(torch.log(predictive_mean_probs + 1e-12), labels, reduction='none')
               
            
            # Accumulate results
            alphas_list.append(batch_adjusted_alphas.cpu())
            member_probs_list.append(batch_member_probs.cpu())
            ensemble_probs_list.append(batch_ensemble_probs.cpu())
            max_probs_list.append(batch_max_probs.cpu())
            member_logits_list.append(batch_model_outputs.cpu())
            predictions_list.append(batch_predictions.cpu())

            epistemic_uncertainty_list.append(batch_epistemic_uncertainty.cpu())
            aleatoric_uncertainty_list.append(batch_aleatoric_uncertainty.cpu())
            aleatoric_median_list.append(batch_aleatoric_median.cpu())
            total_uncertainty_list.append(batch_total_uncertainty.cpu())
            
    return {
        "alphas": torch.cat(alphas_list, dim=0).numpy(),
        "member_probs": torch.cat(member_probs_list, dim=1).numpy(),   # [num_models, num_samples, num_classes]
        "ensemble_probs": torch.cat(ensemble_probs_list, dim=0).numpy(), # [num_samples, num_classes]
        "probabilities": torch.cat(max_probs_list, dim=0).numpy(),       # [num_samples]
        "member_logits": torch.cat(member_logits_list, dim=1).numpy(),   # [num_models, num_samples, num_classes]
        "predictions": torch.cat(predictions_list, dim=0).numpy(),
        "nll": torch.cat(nll_list, dim=0).numpy() if nll_list else None,
        # Uncertainty measures
        "epistemic_uncertainty": torch.cat(epistemic_uncertainty_list, dim=0).numpy(),
        "aleatoric_uncertainty": torch.cat(aleatoric_uncertainty_list, dim=0).numpy(),
        "aleatoric_median": torch.cat(aleatoric_median_list, dim=0).numpy(),
        "total_uncertainty": torch.cat(total_uncertainty_list, dim=0).numpy()
        }
  



  #-----------------------------------------------------------------------------
  # Calibrator for LSTM
  #-----------------------------------------------------------------------------


class CalibratorLSTM():
  
  def __init__(self, deep_ensemble, adjusting_alpha_method="convex_comb", clamping_alphas=False, logits_based_adjustments=True):
    valid_methods = ["convex_comb", "pure_logits", "convex_comb_no_exp", "convex_comb_global"]
    if adjusting_alpha_method.lower() not in valid_methods:
      raise ValueError(f"Invalid adjusting_alpha_method: {adjusting_alpha_method}. Valid methods are {valid_methods}")

    self.ensemble = deep_ensemble
    self.device = self.ensemble.device
    self.aa_method = adjusting_alpha_method.lower()
    self.clamping_alphas = clamping_alphas
    self.logits_based_adjustments = logits_based_adjustments

  ##################################
  # Helper functions BEGIN
  ##################################

  def fit_dirichlet_moment_matching(self, data):
    data = torch.add(data, 1e-6)
    data = data / data.sum(dim=-1, keepdim=True)
    mean_probs = data.mean(dim=0)
    var_probs = data.var(dim=0)
    mean_sum = mean_probs.sum(dim=-1, keepdim=True)
    numerator = mean_sum * (1 - mean_probs).sum(dim=-1, keepdim=True)
    denominator = var_probs.sum(dim=-1, keepdim=True)
    alpha0 = numerator / (denominator + 1e-6) - 1
    alphas = mean_probs * alpha0
    return alphas

  def scaling_alphas(self, alphas):
    assert torch.all(alphas > 0), "All alpha values must be positive."
    if self.clamping_alphas:
      alphas_sorted, _ = torch.sort(alphas, dim=-1)
      alphas_second_min = alphas_sorted[:, 1:2]
      mask = alphas_second_min < 1
      scalar = (1/alphas_second_min).masked_fill(~mask, 1.0)
      alphas_scaled = alphas * scalar
      return alphas_scaled
    else:
      return alphas

  def average_std_logits(self, model_outputs):
    std_logits = torch.var(model_outputs, dim=0)
    mean_std = torch.sqrt(torch.mean(std_logits, dim=-1))
    return mean_std

  def adjust_alphas(self, alphas, epistemic_uncertainty, c2, epi_scalar):
    if self.aa_method == "convex_comb":
      return self.adjust_alphas_CC(alphas, epistemic_uncertainty, c2, epi_scalar)
    elif self.aa_method == "pure_logits":
      return None
    elif self.aa_method == "convex_comb_no_exp":
       return self.adjust_alphas_CC_without_exp(alphas, epistemic_uncertainty, c2, epi_scalar)
    elif self.aa_method == "convex_comb_global":
      return self.adjusting_alphas_CC_global(alphas, epistemic_uncertainty, c2, epi_scalar)
    else:
      raise ValueError(f"Unknown method: {self.aa_method}. Valid options are ['convex_comb','pure_logits', 'convex_comb_no_exp', 'convex_comb_global']")

  def adjust_alphas_CC(self, alphas, epistemic_uncertainty, c2, scalar):
    assert torch.all(alphas > 0), "All alpha values must be positive"
    temp_c2 = torch.exp(c2 * epistemic_uncertainty * scalar)
    rho = (1.0 / temp_c2).unsqueeze(1)
    adjusted_alphas = (1 - rho) * 1 + rho * alphas
    return adjusted_alphas

  def adjust_alphas_CC_without_exp(self, alphas, epistemic_uncertainty, c2, scalar):
    assert torch.all(alphas > 0), "All alpha values must be positive"
    temp_c2 = (c2 * epistemic_uncertainty * scalar)
    rho = (1.0 / (1 + temp_c2)).unsqueeze(1)
    adjusted_alphas = (1 - rho) * 1 + rho * alphas
    return adjusted_alphas

  def adjusting_alphas_CC_global(self, alphas, epistemic_uncertainty, c2, scalar):
    assert torch.all(alphas > 0), "All alpha values must be positive"
    adjusted_alphas = c2 * 1 + (1 - c2) * alphas
    return adjusted_alphas

  def adjust_logits(self, logits, c2):
    mean_logits = logits.mean(dim=0)
    updated_logits = mean_logits + c2 * (logits - mean_logits)
    return updated_logits
  
  #---------------------------------
  # Uncertainty estimation - Based only on Dirichlet
  #---------------------------------

  def mi_dir_monte_carlo(self, alphas, num_samples=1e4):
    num_samples = int(num_samples)
    dirichlet = torch.distributions.Dirichlet(alphas)
    prob_samples = dirichlet.sample([num_samples])
    aleatoric = torch.mean(-torch.sum(prob_samples * torch.log(prob_samples + 1e-8), dim=-1), dim=0)
    aleatoric_median = torch.median(-torch.sum(prob_samples * torch.log(prob_samples + 1e-8), dim=-1), dim=0).values
    total = -torch.sum(torch.mean(prob_samples, dim=0) * torch.log(torch.mean(prob_samples, dim=0) + 1e-8), dim=-1)
    epistemic = total - aleatoric
    return aleatoric, epistemic, total, aleatoric_median
  
  #---------------------------------
  # Uncertainty estimation - partly with Deep Ensemble - used in NLL
  #---------------------------------

  def mi_dir_de_mix(self, member_probs, mean_probs):
    model_entropies = -torch.sum(member_probs * torch.log(member_probs + 1e-8), dim=-1)
    aleatoric = model_entropies.mean(dim=0)
    total = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
    epistemic = total - aleatoric
    return aleatoric, epistemic, total

  ##################################
  # Helper functions END
  ##################################

  #---------------------------------
  # Compute NLL (Fully in Torch)
  #---------------------------------
  def validation_nll(self, val_loader, c1, c2, epi_scalar):
    """
    Compute average NLL over the entire validation set.
    
    Changes:
      - Unpacks (dynamic_inputs, static_inputs, labels) from the loader.
      - Passes both dynamic and static inputs to each model.
    """
    total_loss = 0.0
    total_count = 0
    self.ensemble.eval()

    # Changed: unpack dynamic_inputs and static_inputs
    for dynamic_inputs, static_inputs, labels in val_loader:
      dynamic_inputs = dynamic_inputs.to(self.device)
      static_inputs = static_inputs.to(self.device)
      labels = labels.to(self.device)

      if self.aa_method in ["convex_comb", "convex_comb_no_exp", "convex_comb_global"]:
        with torch.no_grad():
          model_outputs = []
          for model in self.ensemble.models:
            outputs = model(dynamic_inputs, static_inputs)  # Changed: pass both inputs
            scaled_outputs = outputs / c1
            model_outputs.append(scaled_outputs)

          model_outputs = torch.stack(model_outputs)
          member_probs = F.softmax(model_outputs, dim=-1)
          mean_probs = member_probs.mean(dim=0)

          alphas = self.fit_dirichlet_moment_matching(member_probs)
          alphas = self.scaling_alphas(alphas)
        alphas = alphas.to(self.device)
        if not self.logits_based_adjustments:
          _, epistemic_uncertainty, _ = self.mi_dir_de_mix(member_probs, mean_probs)
          adjusted_alphas = self.adjust_alphas(alphas, epistemic_uncertainty, c2, epi_scalar)
        else:
          average_std = self.average_std_logits(model_outputs)
          adjusted_alphas = self.adjust_alphas(alphas, average_std, c2, epi_scalar)
        adjusted_mean_probs = adjusted_alphas / adjusted_alphas.sum(dim=1, keepdim=True)
        log_probs = torch.log(adjusted_mean_probs + 1e-12)
        nll_batch = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
      
      elif self.aa_method == "pure_logits":
        model_outputs = []
        for model in self.ensemble.models:
          outputs = model(dynamic_inputs, static_inputs)  # Changed: pass both inputs
          scaled_outputs = outputs / c1
          model_outputs.append(scaled_outputs)
        model_outputs = torch.stack(model_outputs)
        adjusted_logits = self.adjust_logits(model_outputs, c2)
        member_probs = F.softmax(adjusted_logits, dim=-1)
        mean_probs = member_probs.mean(dim=0)
        log_probs = torch.log(mean_probs + 1e-12)
        nll_batch = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
      else:
        raise ValueError(f"Unknown aa_method: {self.aa_method}")

      total_loss += nll_batch.sum()
      total_count += dynamic_inputs.size(0)
    final_nll = total_loss / total_count
    return final_nll

  #---------------------------------
  # Compute NLL efficiently for grid search
  #---------------------------------
  def precompute_val_outputs(self, val_loader):
    self.ensemble.eval()
    outputs_list = []
    labels_list = []
    with torch.no_grad():
      for dynamic_inputs, static_inputs, labels in val_loader:
        dynamic_inputs = dynamic_inputs.to(self.device)
        static_inputs = static_inputs.to(self.device)
        labels = labels.to(self.device)
        batch_outputs = []
        for model in self.ensemble.models:
          outputs = model(dynamic_inputs, static_inputs)  # Changed: pass both inputs
          batch_outputs.append(outputs)
        batch_outputs = torch.stack(batch_outputs)
        outputs_list.append(batch_outputs)
        labels_list.append(labels)
    model_outputs = torch.cat(outputs_list, dim=1)
    labels = torch.cat(labels_list, dim=0)
    return model_outputs, labels

  def validation_nll_grid_from_precomputed(self, model_outputs, labels, c1, c2, epi_scalar):
    if self.aa_method in ["convex_comb", "convex_comb_no_exp", "convex_comb_global"]:
      scaled_outputs = model_outputs / c1
      member_probs = F.softmax(scaled_outputs, dim=-1)
      mean_probs = member_probs.mean(dim=0)
      alphas = self.fit_dirichlet_moment_matching(member_probs)
      alphas = self.scaling_alphas(alphas)
      if not self.logits_based_adjustments:
        _, epistemic_uncertainty, _ = self.mi_dir_de_mix(member_probs, mean_probs)
        adjusted_alphas = self.adjust_alphas(alphas, epistemic_uncertainty, c2, epi_scalar)
      else:
        average_std = self.average_std_logits(scaled_outputs)
        adjusted_alphas = self.adjust_alphas(alphas, average_std, c2, epi_scalar)
      adjusted_mean_probs = adjusted_alphas / adjusted_alphas.sum(dim=1, keepdim=True)
      log_probs = torch.log(adjusted_mean_probs + 1e-12)
      nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    
    elif self.aa_method == "pure_logits":
      scaled_outputs = model_outputs / c1
      adjusted_logits = self.adjust_logits(scaled_outputs, c2)
      member_probs = F.softmax(adjusted_logits, dim=-1)
      mean_probs = member_probs.mean(dim=0)
      log_probs = torch.log(mean_probs + 1e-12)
      nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    else:
      raise ValueError(f"Unknown aa_method: {self.aa_method}")
    
    final_nll = nll.mean()
    return final_nll

  #---------------------------------
  # GRID SEARCH
  #---------------------------------
  def grid_search_c1_c2_precomputed(self, val_loader, c1_vals, c2_vals, epi_scalar_vals):
    results = []
    best_nll = float('inf')
    best_params = {"c1": None, "c2": None, "epi_scalar": None, "nll_val": None}
    
    model_outputs, labels = self.precompute_val_outputs(val_loader)
    for c1 in c1_vals:
      for c2 in c2_vals:
        for epi_scalar in epi_scalar_vals:
          final_nll = self.validation_nll_grid_from_precomputed(model_outputs, labels, c1, c2, epi_scalar)
          results.append((c1, c2, epi_scalar, final_nll.item()))
          if final_nll.item() < best_nll:
            best_nll = final_nll.item()
            best_params = {"c1": c1, "c2": c2, "epi_scalar": epi_scalar, "nll_val": best_nll}
    return results, best_params

  def grid_search_c1_c2(self, val_loader, c1_vals, c2_vals, epi_scalar_vals):
    results = []
    best_nll = float('inf')
    best_params = {"c1": None, "c2": None, "epi_scalar": None, "nll": None}
    for c1 in c1_vals:
      for c2 in c2_vals:
        for epi_scalar in epi_scalar_vals:
          final_nll = self.validation_nll(val_loader, c1, c2, epi_scalar)
          results.append((c1, c2, epi_scalar, final_nll.item()))
          if final_nll.item() < best_nll:
            best_nll = final_nll.item()
            best_params = {"c1": c1, "c2": c2, "epi_scalar": epi_scalar, "nll_val": best_nll}
    return results, best_params

  def optimize_c1_c2(self, val_loader, c1_vals, epi_scalar_lr_tuples, c2_init, c2_steps=50, tolerance=1e-6, patience=2):
    results = []
    best_nll = float('inf')
    theta_init = np.log(np.exp(c2_init) - 1)
    for c1 in c1_vals:
      for epi_scalar, lr in epi_scalar_lr_tuples:
        theta = torch.tensor(theta_init, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=lr)
        no_improvement = 0
        prev_nll = float('inf')
        for step in range(c2_steps):
          optimizer.zero_grad()
          c2 = F.softplus(theta)
          nll_tensor = self.validation_nll(val_loader, c1, c2, epi_scalar)
          nll_val = nll_tensor.item()
          if abs(prev_nll - nll_val) < tolerance:
            no_improvement += 1
          else:
            no_improvement = 0
          prev_nll = nll_val
          if no_improvement >= patience:
            break
          nll_tensor.backward()
          optimizer.step()
        c2 = F.softplus(theta)
        final_nll = self.validation_nll(val_loader, c1, c2, epi_scalar)
        results.append((c1, c2.item(), epi_scalar, lr, final_nll.item()))
        if final_nll.item() < best_nll:
          best_nll = final_nll.item()
          best_params = {"c1": c1, "c2": c2.item(), "epi_scalar": epi_scalar, "lr": lr, "nll_val": best_nll}
    return results, best_params

  #---------------------------------
  # PREDICTING
  #---------------------------------
  def predict(self, loader, c1_prim, c2_prim, epi_scalar_prim):
    alphas_list = []
    member_probs_list = []
    ensemble_probs_list = []
    max_probs_list = []
    predictions_list = []
    member_logits_list = []
    nll_list = []
    epistemic_uncertainty_list = []
    aleatoric_uncertainty_list = []
    aleatoric_median_list = []
    total_uncertainty_list = []
    labels_list = []

    self.ensemble.eval()
    with torch.no_grad():
      for batch in loader:
        # Changed: check for multi-input batches
        if len(batch) == 3:
          dynamic_inputs, static_inputs, labels = batch
          dynamic_inputs = dynamic_inputs.to(self.device)
          static_inputs = static_inputs.to(self.device)
          labels = labels.to(self.device)
        elif len(batch) == 2:
          dynamic_inputs, static_inputs = batch
          dynamic_inputs = dynamic_inputs.to(self.device)
          static_inputs = static_inputs.to(self.device)
          labels = None
        else:
          raise ValueError("Batch format not recognized. Expected 2 or 3 elements.")

        batch_model_outputs = []
        for model in self.ensemble.models:
          outputs = model(dynamic_inputs, static_inputs)  # Changed: pass both inputs
          scaled_outputs = outputs / c1_prim
          batch_model_outputs.append(scaled_outputs)
        batch_model_outputs = torch.stack(batch_model_outputs)
        if self.aa_method == "pure_logits":
          adjusted_logits = self.adjust_logits(batch_model_outputs, c2_prim)
          batch_member_probs = F.softmax(adjusted_logits, dim=-1)
          batch_ensemble_probs = batch_member_probs.mean(dim=0)
          batch_alphas = self.fit_dirichlet_moment_matching(batch_member_probs)
          batch_adjusted_alphas = self.scaling_alphas(batch_alphas)
        else:
          batch_member_probs = F.softmax(batch_model_outputs, dim=-1)
          batch_mean_probs = batch_member_probs.mean(dim=0)
          batch_alphas = self.fit_dirichlet_moment_matching(batch_member_probs)
          batch_adjusted_alphas = self.scaling_alphas(batch_alphas)
          if not self.logits_based_adjustments:
            _, batch_epistemic_uncertainty, _ = self.mi_dir_de_mix(batch_member_probs, batch_mean_probs)
            batch_adjusted_alphas = self.adjust_alphas(batch_adjusted_alphas, batch_epistemic_uncertainty, c2_prim, epi_scalar_prim)
          else:
            batch_avg_std = self.average_std_logits(batch_model_outputs)
            batch_adjusted_alphas = self.adjust_alphas(batch_adjusted_alphas, batch_avg_std, c2_prim, epi_scalar_prim)
          batch_ensemble_probs = batch_adjusted_alphas / batch_adjusted_alphas.sum(dim=1, keepdim=True)
        
        batch_max_probs, batch_predictions = torch.max(batch_ensemble_probs, dim=1)
        batch_aleatoric_uncertainty, batch_epistemic_uncertainty, batch_total_uncertainty, batch_aleatoric_median = self.mi_dir_monte_carlo(batch_adjusted_alphas)
        if labels is not None:
          batch_nll = -torch.gather(torch.log(batch_ensemble_probs + 1e-12), dim=1, index=labels.unsqueeze(1)).squeeze(1)
          nll_list.append(batch_nll.cpu())
        
        alphas_list.append(batch_adjusted_alphas.cpu())
        member_probs_list.append(batch_member_probs.cpu())
        ensemble_probs_list.append(batch_ensemble_probs.cpu())
        max_probs_list.append(batch_max_probs.cpu())
        member_logits_list.append(batch_model_outputs.cpu())
        predictions_list.append(batch_predictions.cpu())
        
        epistemic_uncertainty_list.append(batch_epistemic_uncertainty.cpu())
        aleatoric_uncertainty_list.append(batch_aleatoric_uncertainty.cpu())
        aleatoric_median_list.append(batch_aleatoric_median.cpu())
        total_uncertainty_list.append(batch_total_uncertainty.cpu())
        if labels is not None:
          labels_list.append(labels.cpu())
        
    return {
        "alphas": torch.cat(alphas_list, dim=0).numpy(),
        "member_probs": torch.cat(member_probs_list, dim=1).numpy(),
        "ensemble_probs": torch.cat(ensemble_probs_list, dim=0).numpy(),
        "probabilities": torch.cat(max_probs_list, dim=0).numpy(),
        "member_logits": torch.cat(member_logits_list, dim=1).numpy(),
        "predictions": torch.cat(predictions_list, dim=0).numpy(),
        "labels": torch.cat(labels_list, dim=0).numpy() if labels_list else None,
        "nll": torch.cat(nll_list, dim=0).numpy() if nll_list else None,
        "epistemic_uncertainty": torch.cat(epistemic_uncertainty_list, dim=0).numpy(),
        "aleatoric_uncertainty": torch.cat(aleatoric_uncertainty_list, dim=0).numpy(),
        "aleatoric_median": torch.cat(aleatoric_median_list, dim=0).numpy(),
        "total_uncertainty": torch.cat(total_uncertainty_list, dim=0).numpy()
    }




class PrecomputedCalibrator:
    def __init__(self, adjusting_alpha_method="convex_comb", clamping_alphas=False, 
                 logits_based_adjustments=True):
        """
        A calibrator that works with precomputed probability outputs.
        
        Args:
            adjusting_alpha_method (str): One of ["convex_comb", "pure_logits", "convex_comb_no_exp", "convex_comb_global"].
            clamping_alphas (bool): Whether to clamp/scale the alphas.
            logits_based_adjustments (bool): Flag to choose between uncertainty based on logits or other measures.
            device (torch.device): The device to run computations on.
        """
        valid_methods = ["convex_comb", "pure_logits", "convex_comb_no_exp", "convex_comb_global"]
        if adjusting_alpha_method.lower() not in valid_methods:
            raise ValueError(f"Invalid adjusting_alpha_method: {adjusting_alpha_method}. Valid methods are {valid_methods}")
        self.aa_method = adjusting_alpha_method.lower()
        self.clamping_alphas = clamping_alphas
        self.logits_based_adjustments = logits_based_adjustments
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ##################################
    # Helper functions BEGIN
    ##################################
    def fit_dirichlet_moment_matching(self, data):
        """
        Fits a Dirichlet distribution via moment matching.
        
        Args:
            data (torch.Tensor): Tensor of shape [num_members, num_samples, num_classes] (probabilities).
        Returns:
            alphas (torch.Tensor): Estimated Dirichlet parameters, shape [num_samples, num_classes].
        """
        data = torch.add(data, 1e-6)
        data = data / data.sum(dim=-1, keepdim=True)
        mean_probs = data.mean(dim=0)  # [num_samples, num_classes]
        var_probs = data.var(dim=0)
        mean_sum = mean_probs.sum(dim=-1, keepdim=True)
        numerator = mean_sum * (1 - mean_probs).sum(dim=-1, keepdim=True)
        denominator = var_probs.sum(dim=-1, keepdim=True)
        alpha0 = numerator / (denominator + 1e-6) - 1
        alphas = mean_probs * alpha0
        return alphas

    def scaling_alphas(self, alphas):
        """
        Optionally scale (clamp) the alphas.
        
        Args:
            alphas (torch.Tensor): [num_samples, num_classes]
        Returns:
            alphas_scaled (torch.Tensor): Scaled alphas.
        """
        assert torch.all(alphas > 0), "All alpha values must be positive."

        if self.clamping_alphas:
            alphas_sorted, _ = torch.sort(alphas, dim=-1) #returns idx
            alphas_second_min = alphas_sorted[:, 1:2]
            mask = alphas_second_min < 1
            scalar = (1/alphas_second_min).masked_fill(~mask, 1.0) #Complement gets 1.0  shape [batch_size]
            alphas_scaled = alphas * scalar
            return alphas_scaled
        #if false do nothing
        else:
            return alphas

    def average_std_logits(self, model_outputs):
        """
        Computes the average standard deviation across models for each sample.
        
        Args:
            model_outputs (torch.Tensor): [num_members, num_samples, num_classes] (logits).
        Returns:
            mean_std (torch.Tensor): [num_samples]
        """
        std_logits = torch.var(model_outputs, dim=0)  # Shape: [batch_size, num_classes]
        mean_std = torch.sqrt(torch.mean(std_logits, dim=-1))  # Shape: [batch_size]
        return mean_std

    def adjust_alphas(self, alphas, uncertainty, c2, epi_scalar):
        """
        Dispatches to one of the alpha adjustment methods.
        
        Args:
            alphas (torch.Tensor): [num_samples, num_classes]
            uncertainty (torch.Tensor): [num_samples] uncertainty measure.
            c2: calibration parameter (float or tensor)
            epi_scalar: scalar applied to uncertainty.
        Returns:
            adjusted_alphas (torch.Tensor)
        """
        if self.aa_method == "convex_comb":
            return self.adjust_alphas_CC(alphas, uncertainty, c2, epi_scalar)
        elif self.aa_method == "pure_logits":
            return None  # No adjustment in pure_logits case.
        elif self.aa_method == "convex_comb_no_exp":
            return self.adjust_alphas_CC_without_exp(alphas, uncertainty, c2, epi_scalar)
        elif self.aa_method == "convex_comb_global":
            return self.adjusting_alphas_CC_global(alphas, uncertainty, c2, epi_scalar)
        else:
            raise ValueError(f"Unknown aa_method: {self.aa_method}")

    def adjust_alphas_CC(self, alphas, epistemic_uncertainty, c2, scalar):
        assert torch.all(alphas > 0), "All alpha values must be positive"
        temp_c2 = torch.exp(c2 * epistemic_uncertainty * scalar)
        rho = (1.0 / temp_c2).unsqueeze(1)
        adjusted_alphas = (1 - rho) * 1 + rho * alphas
        return adjusted_alphas

    def adjust_alphas_CC_without_exp(self, alphas, epistemic_uncertainty, c2, scalar):
        assert torch.all(alphas > 0), "All alpha values must be positive"
        temp_c2 = c2 * epistemic_uncertainty * scalar
        rho = (1.0 / (1 + temp_c2)).unsqueeze(1)
        adjusted_alphas = (1 - rho) * 1 + rho * alphas
        return adjusted_alphas

    def adjusting_alphas_CC_global(self, alphas, epistemic_uncertainty, c2, scalar):
        assert torch.all(alphas > 0), "All alpha values must be positive"
        adjusted_alphas = c2 * 1 + (1 - c2) * alphas
        return adjusted_alphas

    def adjust_logits(self, logits, c2):
        """
        Adjusts logits for the pure_logits method.
        
        Args:
            logits (torch.Tensor): [num_members, num_samples, num_classes]
            c2: calibration parameter.
        Returns:
            updated_logits (torch.Tensor)
        """
        mean_logits = logits.mean(dim=0)
        updated_logits = mean_logits + c2 * (logits - mean_logits)
        return updated_logits
    
    #---------------------------------
    # Uncertainty estimation
    #---------------------------------

    def mi_dir_monte_carlo(self, alphas, num_samples=1e4):
        """
        Monte Carlo approximation of the Dirichlet mutual information.
        
        Args:
            alphas (torch.Tensor): [num_samples, num_classes]
            num_samples (int): Number of samples for MC estimation.
        Returns:
            aleatoric, epistemic, total, aleatoric_median (each [num_samples])
        """
        num_samples = int(num_samples)
        dirichlet = torch.distributions.Dirichlet(alphas)
        prob_samples = dirichlet.sample([num_samples])
        aleatoric = torch.mean(-torch.sum(prob_samples * torch.log(prob_samples + 1e-8), dim=-1), dim=0)
        aleatoric_median = torch.median(-torch.sum(prob_samples * torch.log(prob_samples + 1e-8), dim=-1), dim=0).values
        total = -torch.sum(torch.mean(prob_samples, dim=0) * torch.log(torch.mean(prob_samples, dim=0) + 1e-8), dim=-1)
        epistemic = total - aleatoric
        return aleatoric, epistemic, total, aleatoric_median

    def mi_dir_de_mix(self, member_probs, mean_probs):
        """
        Uncertainty estimation using deep ensemble mixtures.
        
        Args:
            member_probs (torch.Tensor): [num_members, num_samples, num_classes]
            mean_probs (torch.Tensor): [num_samples, num_classes]
        Returns:
            aleatoric, epistemic, total (each [num_samples])
        """
        model_entropies = -torch.sum(member_probs * torch.log(member_probs + 1e-8), dim=-1)
        aleatoric = model_entropies.mean(dim=0)
        total = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        epistemic = total - aleatoric
        return aleatoric, epistemic, total

    
    def mi_with_probs(self, member_probs: torch.Tensor)-> tuple:
        """
        Compute the mutual information using probabilities.
        
        Args:
            member_probs (torch.Tensor): [num_members, num_samples, num_classes]
        Returns:
            aleatoric, epistemic, total (each [num_samples])
        """
        # Compute the mean probabilities across ensemble members.
        mean_probs = member_probs.mean(dim=0) #[num_samples, num_classes]
        total_uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)  # Shape: [num_samples]
        aleatoric = torch.mean(-torch.sum(member_probs * torch.log(member_probs + 1e-8), dim=-1), dim=0)  # Shape: [num_samples]
        epistemic = total_uncertainty - aleatoric
        return aleatoric, epistemic, total_uncertainty

    ##################################
    # Helper functions END
    ##################################


    def validation_nll(self, precomputed_probs, labels, c1, c2, epi_scalar):
        """
        Computes the average negative log-likelihood (NLL) over all samples using precomputed outputs.
        
        Args:
            precomputed_probs (torch.Tensor): [num_members, total_samples, num_classes] (probabilities).
            labels (torch.Tensor): [total_samples] true labels.
            c1 (float): Temperature scaling factor.
            c2: Calibration parameter.
            epi_scalar (float): Scalar for uncertainty adjustment.
        Returns:
            final_nll (torch.Tensor): A scalar tensor representing the average NLL.
        """
        #get logits - non-unique
        precomputed_outputs = torch.log(precomputed_probs + 1e-12)

        precomputed_outputs = precomputed_outputs.to(self.device)
        labels = labels.to(self.device)
        if self.aa_method in ["convex_comb", "convex_comb_no_exp", "convex_comb_global"]:
            scaled_outputs = precomputed_outputs / c1
            member_probs = F.softmax(scaled_outputs, dim=-1)
            mean_probs = member_probs.mean(dim=0)
            alphas = self.fit_dirichlet_moment_matching(member_probs)
            alphas = self.scaling_alphas(alphas)
            if not self.logits_based_adjustments:
                _, epistemic_uncertainty, _ = self.mi_dir_de_mix(member_probs, mean_probs)
                adjusted_alphas = self.adjust_alphas(alphas, epistemic_uncertainty, c2, epi_scalar)
            else:
                average_std = self.average_std_logits(scaled_outputs)
                adjusted_alphas = self.adjust_alphas(alphas, average_std, c2, epi_scalar)
            adjusted_mean_probs = adjusted_alphas / adjusted_alphas.sum(dim=1, keepdim=True)
            log_probs = torch.log(adjusted_mean_probs + 1e-12)
            nll_batch = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        elif self.aa_method == "pure_logits":
            scaled_outputs = precomputed_outputs / c1
            adjusted_logits = self.adjust_logits(scaled_outputs, c2)
            member_probs = F.softmax(adjusted_logits, dim=-1)
            mean_probs = member_probs.mean(dim=0)
            log_probs = torch.log(mean_probs + 1e-12)
            nll_batch = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError(f"Unknown aa_method: {self.aa_method}")
        final_nll = nll_batch.mean()
        return final_nll

    def grid_search_c1_c2_precomputed(self, precomputed_probs, labels, c1_vals, c2_vals, epi_scalar_vals):
        
        results = []
        best_nll = float('inf')
        best_params = {"c1": None, "c2": None, "epi_scalar": None, "nll_val": None}
        for c1 in c1_vals:
            for c2 in c2_vals:
                for epi_scalar in epi_scalar_vals:
                    final_nll = self.validation_nll(precomputed_probs, labels, c1, c2, epi_scalar)
                    results.append((c1, c2, epi_scalar, final_nll.item()))
                    if final_nll.item() < best_nll:
                        best_nll = final_nll.item()
                        best_params = {"c1": c1, "c2": c2, "epi_scalar": epi_scalar, "nll_val": best_nll}
        return results, best_params

    ##---------------------------------
    ## Faster implementation of validation nll and grid search
    ##---------------------------------

    def optimized_validation_nll(self, precomputed_logits, labels, c1, c2, epi_scalar, eps=1e-12):
        """
        Optimized version of validation_nll.
        precomputed in grid_search_c1_c2_precomputed_optimized
        precomputed_logits: precomputed torch.log(precomputed_probs + eps), shape [num_members, num_samples, num_classes]
        """
        
        labels = labels.to(self.device)
        if self.aa_method in ["convex_comb", "convex_comb_no_exp", "convex_comb_global"]:
            scaled_outputs = precomputed_logits / c1
            member_probs = F.softmax(scaled_outputs, dim=-1)
            mean_probs = member_probs.mean(dim=0)
            alphas = self.fit_dirichlet_moment_matching(member_probs)
            alphas = self.scaling_alphas(alphas)
            if not self.logits_based_adjustments:
                _, epistemic_uncertainty, _ = self.mi_dir_de_mix(member_probs, mean_probs)
                adjusted_alphas = self.adjust_alphas(alphas, epistemic_uncertainty, c2, epi_scalar)
            else:
                average_std = self.average_std_logits(scaled_outputs)
                adjusted_alphas = self.adjust_alphas(alphas, average_std, c2, epi_scalar)
            adjusted_mean_probs = adjusted_alphas / adjusted_alphas.sum(dim=1, keepdim=True)
            log_probs = torch.log(adjusted_mean_probs + eps)
            nll_batch = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        elif self.aa_method == "pure_logits":
            scaled_outputs = precomputed_logits / c1
            adjusted_logits = self.adjust_logits(scaled_outputs, c2)
            member_probs = F.softmax(adjusted_logits, dim=-1)
            mean_probs = member_probs.mean(dim=0)
            log_probs = torch.log(mean_probs + eps)
            nll_batch = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError(f"Unknown aa_method: {self.aa_method}")
        final_nll = nll_batch.mean()
        return final_nll

    def grid_search_c1_c2_precomputed_optimized(self, precomputed_probs, labels, c1_vals, c2_vals, epi_scalar_vals, eps=1e-12):
        """
        Optimized grid search using precomputed logits.
        """
        # Precompute the logs once and send to the device.
        precomputed_logits = torch.log(precomputed_probs + eps).to(self.device)
        results = []
        best_nll = float('inf')
        best_params = {"c1": None, "c2": None, "epi_scalar": None, "nll_val": None}
        for c1 in c1_vals:
            for c2 in c2_vals:
                for epi_scalar in epi_scalar_vals:
                    final_nll = self.optimized_validation_nll(precomputed_logits, labels, c1, c2, epi_scalar, eps=eps)
                    nll_val = final_nll.item()
                    results.append((c1, c2, epi_scalar, nll_val))
                    if nll_val < best_nll:
                        best_nll = nll_val
                        best_params = {"c1": c1, "c2": c2, "epi_scalar": epi_scalar, "nll_val": best_nll}
        return results, best_params

    def grid_search_c1_c2_precomputed_coarse_to_fine(self, precomputed_probs, labels, c1_coarse, c2_coarse, epi_scalar_vals, eps=1e-12):
        """
        Coarse-to-fine grid search using the optimized grid search function.
        First, perform grid search on a coarse grid, then refine the search around the best coarse parameters.
        """
        # Stage 1: Coarse grid search
        results_coarse, best_params_coarse = self.grid_search_c1_c2_precomputed_optimized(precomputed_probs, labels, 
                                                    c1_coarse, c2_coarse, epi_scalar_vals, eps=eps)
        
        c1_best = best_params_coarse["c1"]
        c2_best = best_params_coarse["c2"]
        
        # Define a fine grid around the best coarse values.
        c1_fine =np.linspace(max((c1_best- 0.2 * c1_best), min(c1_coarse)), (c1_best+ 0.2* c1_best), 50)
        c2_fine = np.linspace(max((c2_best- 0.2 * c2_best), min(c2_coarse)), (c2_best+ 0.2* c2_best), 50)
        
        # Stage 2: Fine grid search over the refined grid.
        results_fine, best_params_fine = self.grid_search_c1_c2_precomputed_optimized(precomputed_probs, labels, 
                                                c1_fine, c2_fine, epi_scalar_vals, eps=eps)
        
        return results_fine, best_params_fine
    
    ##---------------------------------
    # Ensemble selection methods
    ##---------------------------------

    #------ START HELPER FUNCTION ------

    def nll_at_T(self, logits, labels, T, eps=1e-12):
        """
        Compute mean NLL at a given temperature T.
        
        Args:
            logits (torch.Tensor): [num_samples, num_classes]
            labels (torch.Tensor): [num_samples]
            T (float): Temperature.
        Returns:
            nll_mean (float): Average Negative log-likelihood.
            temp_scaled_probs (torch.Tensor): Softmax probabilities after temperature scaling.
        """
        scaled_logits = logits / T
        temp_scaled_probs = F.softmax(scaled_logits, dim=-1)
        nll = -torch.gather(torch.log(temp_scaled_probs + eps), 1, labels.unsqueeze(1)).squeeze(1)
        return nll.mean().item(), temp_scaled_probs

    def compute_ensemble_nll(self, indices, member_probs, labels, eps=1e-12):
        """
        Compute the NLL for a subset of ensemble members.

        Args:
            OPTIONAL indices (list): Indices of selected ensemble members.
            member_probs (torch.Tensor): [num_members, num_samples, num_classes]
            labels (torch.Tensor): [num_samples]
            eps (float): Small value to avoid log(0).
        
        """
        if indices is None:
            ensemble_probs = member_probs.mean(dim=0)  # [num_samples, num_classes]
        else:
            ensemble_probs = member_probs[indices].mean(dim=0)  # [num_samples, num_classes]
        #unscaled so T = 1
        ensemble_logits = torch.log(ensemble_probs + eps)
        nll, _ = self.nll_at_T(ensemble_logits, labels, T = 1.0, eps=eps)
        return nll
    
    def find_optimal_temperature(self,
                                 logits:       torch.Tensor,
                                 labels:       torch.Tensor,
                                 temperatures: list[float],
                                 eps:          float = 1e-12
                                ) -> float:
        """
        Args:
            logits:       [num_samples, num_classes]
            labels:       [num_samples]
            temperatures: list of floats
            eps:          small value to avoid log(0)
        Returns:
            best_T:      optimal temperature
        """

        if logits.ndim != 2:
            raise ValueError(f"Expected 2D, got {logits.shape}")

        nlls = []
        for T in temperatures:
            nll, _ = self.nll_at_T(logits, labels, T, eps)
            nlls.append(nll)
        return temperatures[np.argmin(nlls)] # best_T

    def temperature_list(self,
                         member_probs: torch.Tensor,
                         labels:       torch.Tensor,
                         temperatures: list[float],
                         eps:          float = 1e-12
                        ) -> list[int]:
        """
        Args:
            member_probs: [num_members, num_samples, num_classes]
            labels:       [num_samples]
            temperatures: list of floats
            eps:          small value to avoid log(0)
        Returns:
            sorted_idx:   list of indices of members sorted by NLL
        """
        num_members = member_probs.shape[0]
        # precompute logits once per member
        member_logits = torch.log(member_probs + eps)  # [M,N,C]

        member_nlls = []
        for i in range(num_members):
            # input of shape [N,C]
            best_T = self.find_optimal_temperature(member_logits[i],
                                                   labels,
                                                   temperatures, eps)
            nll, _ = self.nll_at_T(member_logits[i], labels, best_T, eps)
            member_nlls.append(nll)

        sorted_idx = sorted(range(num_members), key=lambda i: member_nlls[i])
        return sorted_idx
    
    def calibrate_then_pool_temperature(self, member_probs: torch.Tensor, 
                                       labels: torch.Tensor, 
                                       temperatures: list[float], 
                                       eps: float = 1e-12) -> tuple[torch.Tensor, list[float]]:
        """
        Calibrate each ensemble member individually with temperature scaling, then pool the results.
        
        Args:
            member_probs: [num_members, num_samples, num_classes] - probabilities from ensemble members
            labels: [num_samples] - true labels for validation
            temperatures: list of candidate temperatures to try
            eps: small value to avoid log(0)
            
        Returns:
            pooled_probs: [num_samples, num_classes] - averaged calibrated probabilities
            member_temperatures: list of optimal temperatures for each member
        """
        num_members = member_probs.shape[0]
        member_temperatures = []
        calibrated_member_probs = []
        
        # Convert probabilities to logits once
        member_logits = torch.log(member_probs + eps)  # [num_members, num_samples, num_classes]
        
        # Find optimal temperature for each member individually
        for i in range(num_members):
            best_temp = self.find_optimal_temperature(member_logits[i], labels, temperatures, eps)
            member_temperatures.append(best_temp)
            
            # Apply temperature scaling to this member
            scaled_logits = member_logits[i] / best_temp
            calibrated_probs = F.softmax(scaled_logits, dim=-1)
            calibrated_member_probs.append(calibrated_probs)
        
        # Stack and average the calibrated probabilities
        calibrated_member_probs = torch.stack(calibrated_member_probs)  # [num_members, num_samples, num_classes]
        pooled_probs = calibrated_member_probs.mean(dim=0)  # [num_samples, num_classes]
        
        return pooled_probs, member_temperatures
  
    #------ END HELPER FUNCTION ------



    def top_n_ensemble(self, member_probs, labels, N, metric_name="nll", eps=1e-12):
        """
        Simple top N ensemble based on individual NLLs
        """
        if not (torch.is_tensor(member_probs) and torch.is_tensor(labels)):
            raise ValueError('Invalid data type as input')
    
        num_members = member_probs.shape[0]
        if N > num_members:
            N = num_members
        
        nlls = []
        for i in range(num_members):
            nll = self.compute_ensemble_nll([i], member_probs, labels, eps=eps)
            nlls.append(nll)
        
        #sort by nll
        sorted_indices = sorted(range(num_members), key=lambda i: nlls[i])
        selected = sorted_indices[:N]  # Select the top N models based on NLL
        selected_nlls = [nlls[i] for i in selected]
        print(f"TOP {N} ensemble selected: {selected} with NLL = {selected_nlls}")
        return selected, selected_nlls
       
    

    def greedy_ensemble(self, member_probs, labels, m, metric_name="nll", no_resample=True, eps=1e-12 ):
        """
        Greedy ensemble just like the paper

        Args:
            member_probs shape [num_members, num_samples, num_classes] 
            labels shape [num_samples]
            m (int): Number of ensemble members to select.
            metric_name (str): The metric to use for evaluating the ensemble.
            no_resample (bool): If True, do not resample the selected members.
            eps (float): Small value to avoid log(0).
        Returns:
            selected (list of int): The indices (with respect to member_probs) of the selected ensemble members.
            ensemble_metrics (list of float): The ensemble metric after each member
        """
        if not (torch.is_tensor(member_probs) and torch.is_tensor(labels)):
            raise ValueError('Invalid data type as input')
    
        num_members = member_probs.shape[0]
        selected = []       # List to hold the indices of selected models.
        remaining = set(range(num_members))
        ensemble_metrics = []  # To record the metric after each addition.
    
        def compute_metric(indices):
            # Average the probabilities of the models in 'indices'
            ensemble_probs = member_probs[list(indices)].mean(dim=0)  # Shape: [num_samples, num_classes]
            if metric_name == "nll":
                # Negative log-likelihood.
                nll = -torch.gather(torch.log(ensemble_probs + eps), 1, labels.unsqueeze(1)).squeeze(1)
                return nll.mean().item()
            elif metric_name == "error":
                # Classification error rate.
                preds = ensemble_probs.argmax(dim=1)
                error_rate = (preds != labels).float().mean().item()
                return error_rate
            elif metric_name == "relative_absolute_error":
                # Relative absolute error: |pred - true|/clamp(|true|,min=1)
                preds = ensemble_probs.argmax(dim=1).float()
                rel_abs_error = (torch.abs(preds - labels) / torch.clamp(torch.abs(labels), min=1.0)).mean().item()
                return rel_abs_error
            else:
                raise ValueError(f"Unknown metric_name: {metric_name}")
    
        # Greedy forward selection: add one candidate at each iteration.
        for i in range(m):
            best_candidate = None
            best_metric_value = float('inf')
            # Evaluate each candidate in the remaining set.
            for candidate in remaining:
                candidate_set = selected + [candidate]
                candidate_metric = compute_metric(candidate_set)
                if candidate_metric < best_metric_value:
                    best_metric_value = candidate_metric
                    best_candidate = candidate
            # Add the best candidate
            selected.append(best_candidate)
            # If no resampling, remove it from the pool.
            if no_resample:
                remaining.remove(best_candidate)
            ensemble_metrics.append(best_metric_value)
        print(f"GREEDY ensemble selected: {selected} with metrics = {ensemble_metrics}") 
        return selected, ensemble_metrics
    
    def greedy_ensemble_5(self, member_probs, labels, m=5, eps=1e-12, max_iter = 50):
        """
        Greedy ensemble that stops at m unique members (allows resampling)

        Args:
            member_probs shape [num_members, num_samples, num_classes] 
            labels shape [num_samples]
            m (int): Number of unique ensemble members to select.
            eps (float): Small value to avoid log(0).
        Returns:
            selected (list of int): The indices (with respect to member_probs) of the selected ensemble members.
                                   May contain duplicates but will have at most m unique members.
        """
        if not (torch.is_tensor(member_probs) and torch.is_tensor(labels)):
            raise ValueError('Invalid data type as input')
        num_members = member_probs.shape[0]
        selected = []       # List to hold all selected indices (may have duplicates)
        unique_selected = set()  # Track unique members separately
        remaining = set(range(num_members))
        ensemble_metrics = []  # To record the metric after each addition.
    
        # Greedy forward selection: add one candidate at each iteration.
        for iteration in range(max_iter):
            # Stop when we have m unique members
            if len(unique_selected) >= m:
                break
            best_candidate = None
            best_metric_value = float('inf')
            # Evaluate each candidate in the remaining set.
            for candidate in remaining:
                candidate_set = selected + [candidate]
                candidate_ensemble_probs = member_probs[candidate_set].mean(dim=0)  # Shape: [num_samples, num_classes]
                nll = -torch.gather(torch.log(candidate_ensemble_probs + eps), 1, labels.unsqueeze(1)).squeeze(1)
                candidate_metric = nll.mean().item()          # nll_mean as metric

                if candidate_metric < best_metric_value:
                    best_metric_value = candidate_metric
                    best_candidate = candidate
            # Add the best candidate
            selected.append(best_candidate)
            unique_selected.add(best_candidate)  # Track unique members
            ensemble_metrics.append(best_metric_value)
        print(f"GREEDY 5 ensemble selected: {selected} with {len(unique_selected)} unique members and metrics = {ensemble_metrics}") 
        return selected
    

    def greedy_ensemble_calibrated_once(self, member_probs, labels, m, init_N= 5, no_resample = True, tolerance = 3, eps=1e-12, 
                                              c1_vals = np.linspace(0.5, 1.5, 50),
                                              c2_vals = np.linspace(0.5, 1.5, 50), 
                                              epi_scalar_vals = np.array([1.0])):
        """
        Initial N and calibrates once

        """
    
        #ensure m >= init_N
        if init_N > m:
            raise ValueError('Initial ensemble must be smaller than m')
        # Ensure data are tensor.
        if not (torch.is_tensor(member_probs) and torch.is_tensor(labels)):
           raise ValueError('member_probs and labels must both be tensors')

    
        num_members = member_probs.shape[0]

        # 1. Get the individual nll of temp scaled probs
        sorted_indices = self.temperature_list(member_probs, labels, c1_vals)

        init_N = min(init_N, num_members)
        candidate_pool = sorted_indices[:init_N] # best init_N members
        #print(f"Initial candidate pool (top {init_N} models): {candidate_pool}")
    
        selected = candidate_pool.copy()
        # We will no longer used the temp scaled probs, but the original probs
        candidate_probs = member_probs[selected]  # [init_N, num_samples, num_classes]
        remaining = set(range(num_members)) - set(candidate_pool)
        #calibrate
        _, best_params = self.grid_search_c1_c2_precomputed_coarse_to_fine(candidate_probs, labels,
                                                          c1_vals, c2_vals, epi_scalar_vals)
        c1_prim = best_params['c1']
        c2_prim = best_params['c2']
        epi_scalar_prim = best_params['epi_scalar']
        current_nll = best_params['nll_val']
        #print(f"Completed calibration ({self.aa_method}) for initial {candidate_pool} with NLL {current_nll} and params {best_params}")
        ensemble_nlls = [current_nll]
    
        # 5. Greedy forward selection: add candidates only if they improve (i.e. lower) the NLL.
        counter = 0
        while len(selected) < m:
            best_candidate = None
            best_nll = float('inf')
            for candidate in remaining:
                candidate_set = selected + [candidate]
                candidate_result = self.predict(member_probs[candidate_set], c1_prim=c1_prim, c2_prim=c2_prim,
                                                epi_scalar_prim=epi_scalar_prim, labels=labels, eps=eps)

                candidate_nll = candidate_result["nll"].mean()

                if candidate_nll < best_nll:
                    best_nll = candidate_nll
                    best_candidate = candidate
            if best_nll < current_nll:
                selected.append(best_candidate)
                if no_resample:
                    remaining.remove(best_candidate)
                ensemble_nlls.append(best_nll)
                current_nll = best_nll
                #print(f"Added candidate {best_candidate}, ensemble {selected}: NLL = {best_nll:.4f}")
                #reset counter
                counter = 0
            else:
                counter += 1
                if counter >= tolerance:
                    #print("Stopping early.")
                    break
        print(f"Greedy enemble SINGLE calibration; selected {selected} with NLL = {ensemble_nlls}")
        return selected, ensemble_nlls


    def greedy_ensemble_recalibrated(self, member_probs, labels, m, init_N= 5, no_resample = True, tolerance = 3, eps=1e-12, 
                                              c1_vals = np.linspace(0.5, 1.5, 50),
                                              c2_vals = np.linspace(0.5, 1.5, 50), 
                                              epi_scalar_vals = np.array([1.0])):
        """
        Recalibrates at every step in the while loop and starts with temp scaled initial N

        """
        # Ensure data are tensor.
        if not (torch.is_tensor(member_probs) and torch.is_tensor(labels)):
            raise ValueError('member_probs and labels must both be tensors')

    
        num_members = member_probs.shape[0]
        init_N = min(init_N, num_members)
        # 1. Get the individual nll of temp scaled probs
        sorted_indices = self.temperature_list(member_probs, labels, c1_vals)
        selected = sorted_indices[:init_N] # best init_N members - indices
        remaining = set(range(num_members)) - set(selected)

        
        #add to PSEUDO nll to list
        current_nll = self.compute_ensemble_nll(selected, member_probs, labels, eps=eps) 
        ensemble_nlls = [current_nll]

        # Greedy forward selection: add candidates only if they improve (i.e. lower) the NLL.
        counter = 0
        while len(selected) < m and remaining:
            #Get optimal hyperparams for nll calc
            candidate_probs = member_probs[selected]  
            _, current_params = self.grid_search_c1_c2_precomputed_coarse_to_fine(candidate_probs, labels,
                                                          c1_vals, c2_vals, epi_scalar_vals)
            c1_prim = current_params['c1']
            c2_prim = current_params['c2']
            epi_scalar_prim = current_params['epi_scalar']

            best_candidate = None
            best_nll = float('inf')

            for candidate in remaining:
                candidate_set = selected + [candidate]
                candidate_probs = member_probs[candidate_set] # no longer using temp_scaled probs
                # cant re calibrate every step so we just compute the nll
                candidate_result = self.predict(candidate_probs, c1_prim=c1_prim, c2_prim=c2_prim,
                                                epi_scalar_prim=epi_scalar_prim, labels=labels, eps=eps)

                candidate_nll = candidate_result["nll"].mean().item() #overkill

                if candidate_nll < best_nll:
                    best_nll = candidate_nll
                    best_candidate = candidate
            if best_nll < current_nll:
                selected.append(best_candidate)
                if no_resample:
                    remaining.remove(best_candidate)
                ensemble_nlls.append(best_nll)
                current_nll = best_nll
                #print(f"Added candidate {best_candidate}, ensemble {selected}: NLL = {best_nll:.4f}")
                #reset counter
                counter = 0
            else:
                counter += 1
                if counter >= tolerance:
                    #print("Stopping early.")
                    break
        print(f"Greedy REcalibrated ensemble; selected {selected} with NLL = {ensemble_nlls}")
        return selected, ensemble_nlls

    def greedy_ensemble_calibrated_subset(self, member_probs, labels, m, subset_size=None, seed=None,
                                        no_resample=True, tolerance=3, eps=1e-12,
                                        c1_vals=np.linspace(0.5, 1.5, 50),
                                        c2_vals=np.linspace(0.5, 1.5, 50),
                                        epi_scalar_vals=np.array([1.0])):
        """  
        Greedy and recalibrated without initial N - can enforece subset of M
        """
        # Ensure data are tensors.
        if not (torch.is_tensor(member_probs) and torch.is_tensor(labels)):
            raise ValueError('member_probs and labels must both be tensors')
            
        num_members = member_probs.shape[0]
        m = min(m, num_members)
        # Determine the candidate pool:
        if seed is not None and subset_size is not None:
            subset_size = min(subset_size, num_members)
          
            np.random.seed(seed)
            candidate_pool = np.random.choice(np.arange(num_members), size=min(subset_size, num_members), replace=False).tolist()
          
        else:
            # Use all ensemble members.
            candidate_pool = list(range(num_members))

        #cant start empty
        individual_nlls = []
        for i in candidate_pool:
            probs = member_probs[i]  # [num_samples, num_classes]
            nll = -torch.gather(torch.log(probs + eps), 1, labels.unsqueeze(1)).squeeze(1)
            individual_nlls.append(nll.mean().item())
        # best initial member
        best_initial = candidate_pool[np.argmin(individual_nlls)] 
        selected = [best_initial]
        remaining = set(candidate_pool) - set(selected)

        current_nll = self.compute_ensemble_nll(selected, member_probs, labels, eps=eps)
        ensemble_nlls = [current_nll]
        no_improve_counter = 0

        # Greedy selection: continue until the ensemble reaches m models or we run out of candidates.
        while len(selected) < m and remaining:
            #to avoid moment matching error - NOTE approximation
            if len(selected) < 2:
                c2_prim = float(np.percentile(c2_vals, 15))
                c1_prim = 1.1
                epi_scalar_prim = 1.0
            else:
                candidate_probs = member_probs[selected]  
                _, current_params = self.grid_search_c1_c2_precomputed_coarse_to_fine(candidate_probs, labels,
                                                          c1_vals, c2_vals, epi_scalar_vals)
                c1_prim = current_params['c1']
                c2_prim = current_params['c2']
                epi_scalar_prim = current_params['epi_scalar']

            best_candidate = None
            best_nll = float('inf')
          
            # Evaluate each candidate from the remaining subset.
            for candidate in remaining:
                candidate_set = selected + [candidate]
                candidate_probs = member_probs[candidate_set]  # shape: [ensemble_size, num_samples, num_classes]
            
                candidate_result = self.predict(candidate_probs, c1_prim=c1_prim, c2_prim=c2_prim,
                                                epi_scalar_prim=epi_scalar_prim, labels=labels, eps=eps)

                candidate_nll = candidate_result["nll"].mean().item() #overkill

                if candidate_nll < best_nll:
                    best_nll = candidate_nll
                    best_candidate = candidate
            
            if best_candidate is not None and best_nll < current_nll:
                selected.append(best_candidate)
                if no_resample:
                    remaining.remove(best_candidate)
                ensemble_nlls.append(best_nll)
                current_nll = best_nll
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                if no_improve_counter >= tolerance:
                    break
        print(f"Greedy recalibrated SUBSET ensemble; selected {selected} with NLL = {ensemble_nlls}")
        return selected, ensemble_nlls
       


    ###---------------------------------
    # PREDICTING
    ###---------------------------------
        


    def predict(self, precomputed_probs, c1_prim, c2_prim, epi_scalar_prim, labels=None, eps=1e-12):
        # Convert precomputed probabilities to pseudo-logits (non-unique up to an additive constant).
        precomputed_outputs = torch.log(precomputed_probs + 1e-12)
        precomputed_outputs = precomputed_outputs.to(self.device)
    
        result = {}
    
        if self.aa_method in ["convex_comb", "convex_comb_no_exp", "convex_comb_global"]:
            scaled_outputs = precomputed_outputs / c1_prim
            member_probs = F.softmax(scaled_outputs, dim=-1)  # [num_members, num_samples, num_classes]
            mean_probs = member_probs.mean(dim=0)              # [num_samples, num_classes]
            # Fit Dirichlet parameters (alphas) via moment matching.
            alphas = self.fit_dirichlet_moment_matching(member_probs)  # [num_samples, num_classes]
            alphas = self.scaling_alphas(alphas)
        
            if not self.logits_based_adjustments:
                # Use deep ensemble mixture for uncertainty.
                _, epistemic_uncertainty, _ = self.mi_dir_de_mix(member_probs, mean_probs)
                adjusted_alphas = self.adjust_alphas(alphas, epistemic_uncertainty, c2_prim, epi_scalar_prim)
            else:
                average_std = self.average_std_logits(scaled_outputs)
                adjusted_alphas = self.adjust_alphas(alphas, average_std, c2_prim, epi_scalar_prim)
        
            calibrated_probs = adjusted_alphas / adjusted_alphas.sum(dim=1, keepdim=True)  # [num_samples, num_classes]
            max_probs, preds = torch.max(calibrated_probs, dim=1)
        
            result["predictions"] = preds.cpu().numpy()
            result["probabilities"] = max_probs.cpu().numpy()
            result["ensemble_probs"] = calibrated_probs.cpu().numpy()
            result["alphas"] = adjusted_alphas.cpu().numpy()
            
        elif self.aa_method == "pure_logits":
            scaled_outputs = precomputed_outputs / c1_prim
            adjusted_logits = self.adjust_logits(scaled_outputs, c2_prim)
            member_probs = F.softmax(adjusted_logits, dim=-1)
            alphas = self.fit_dirichlet_moment_matching(member_probs)
            adjusted_alphas = self.scaling_alphas(alphas)
            mean_probs = member_probs.mean(dim=0)  # [num_samples, num_classes]
            max_probs, preds = torch.max(mean_probs, dim=1)
        
            result["predictions"] = preds.cpu().numpy()
            result["probabilities"] = max_probs.cpu().numpy()
            result["ensemble_probs"] = mean_probs.cpu().numpy()
            result["alphas"] = adjusted_alphas.cpu().numpy()

        else:
            raise ValueError(f"Unknown aa_method: {self.aa_method}")
        
        #compute total, aleatoric and epistemic uncertainty
        # with alphas
        aleatoric_mc, epistemic_mc, total_mc, _ = self.mi_dir_monte_carlo(adjusted_alphas)
        #with probs
        aleatoric_p, epistemic_p, total_p = self.mi_with_probs(member_probs)
        result["aleatoric_mc"] = aleatoric_mc.cpu().numpy()
        result["epistemic_mc"] = epistemic_mc.cpu().numpy()
        result["total_mc"] = total_mc.cpu().numpy()
        result["aleatoric_p"] = aleatoric_p.cpu().numpy()
        result["epistemic_p"] = epistemic_p.cpu().numpy()
        result["total_p"] = total_p.cpu().numpy()

        # Compute NLL if labels are provided.
        if labels is not None:
            labels = labels.to(self.device) if torch.is_tensor(labels) else torch.tensor(labels).to(self.device)
            nll = -torch.log(mean_probs.gather(1, labels.unsqueeze(1)) + eps).squeeze(1)
            result["nll"] = nll.cpu().numpy()
    
        return result
