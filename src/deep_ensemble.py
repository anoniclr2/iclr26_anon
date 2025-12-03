# NEW ensemble
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepEnsemble(nn.Module):
    def __init__(self, num_models, model_class, lr, weight_decay=0, seed=None, **model_kwargs):
        """
        Initialize the ensemble with a number of models.

        Args:
            num_models (int): Number of models in the ensemble.
            model_class (class): The class of the individual models.
            lr (float): Learning rate for optimizers.
            weight_decay (float): Weight decay for optimizers.
            seed (int, optional): Seed for reproducible model initialization. If None, no seed is set.
            model_kwargs: Additional arguments to pass to the model class.
        """
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([])
        for i in range(num_models):
            # Set seed for reproducible model initialization
            if seed is not None:
                torch.manual_seed(seed + i)  # Different seed per model for diversity
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed + i)
            self.models.append(model_class(**model_kwargs))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Initialize separate optimizers for each model.
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                           for model in self.models]

    ##################################
    # Training
    ##################################

    def train_model(self, trainloader, validationloader, criterion, num_epochs, epsilon,
                     adversarial_training=True, 
                     early_stopping = True, patience=3, min_delta=1e-6):
        history = {'train_losses': [], 'train_accuracies': [], 'val_losses': [], 'val_accuracies': []}

        # For individual early stopping, track these per model:
        best_val_losses = [float('inf')] * self.num_models
        patience_counters = [0] * self.num_models
        active_models = [True] * self.num_models
        best_model_states = [model.state_dict() for model in self.models]  # CHANGE: initialize best states

        for epoch in range(num_epochs):
            self.train()  # Set all models to training mode
            running_loss = 0.0
            num_correct_train = 0
            total_train_samples = 0

            # --- TRAINING LOOP ---
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                ensemble_outputs = []

                for m, model in enumerate(self.models):
                    # Skip training if this model has been early stopped.
                    if not active_models[m]:
                        with torch.no_grad():
                            dummy_out = model(inputs)
                            ensemble_outputs.append(dummy_out)  # Use dummy output for aggregation
                        # Optionally print a message (can be commented out in production)
                        # print("Member model", m, "has already been stopped")
                        continue

                    optimizer = self.optimizers[m]
                    optimizer.zero_grad()

                    if adversarial_training:
                        perturbed_inputs = self.fgsm_attack(model, inputs, labels, criterion, epsilon)
                        combined_inputs = torch.cat([inputs, perturbed_inputs], dim=0)
                        combined_labels = torch.cat([labels, labels], dim=0)
                    else:
                        combined_inputs = inputs
                        combined_labels = labels
                    # forward + backward + optimize
                    outputs = model(combined_inputs)
                    loss = criterion(outputs, combined_labels)
                    loss.backward()
                    optimizer.step()

                    # Use only the clean inputs' outputs for ensemble aggregation.
                    ensemble_outputs.append(outputs[:len(inputs)])

                # Aggregate predictions for the batch
                ensemble_outputs = torch.stack(ensemble_outputs)  # shape: [num_models, batch_size, num_classes]
                averaged_logits = ensemble_outputs.mean(dim=0)  # shape: [batch_size, num_classes]
                #only used for monitoring
                loss_ensemble = criterion(averaged_logits, labels)
                running_loss += loss_ensemble.item() * inputs.size(0)

                ensemble_probs = torch.softmax(ensemble_outputs, dim=-1).mean(dim=0)  # shape: [batch_size, num_classes]
                preds = torch.argmax(ensemble_probs, dim=1)  # shape: [batch_size]
                num_correct_train += (preds == labels).sum().item()
                total_train_samples += labels.size(0)

            # Compute epoch-level training metrics
            train_loss = running_loss / (total_train_samples)
            train_accuracy = 100.0 * num_correct_train / total_train_samples
            history['train_losses'].append(train_loss)
            history['train_accuracies'].append(train_accuracy)

            # --- VALIDATION LOOP ---
            self.eval()
            running_val_loss = 0.0
            num_correct_val = 0
            total_val_samples = 0
            individual_val_losses = [0.0] * self.num_models  # Accumulate individual model losses
            individual_val_counts = [0] * self.num_models

            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # For ensemble aggregation:
                    ensemble_outputs = []
                    for m , model in enumerate(self.models):
                        outputs = model(inputs)
                        ensemble_outputs.append(outputs)
                        # calc loss if active
                        if active_models[m]:
                            loss_model = criterion(outputs, labels)
                            individual_val_losses[m] += loss_model.item() * inputs.shape[0]
                            individual_val_counts[m] += inputs.shape[0]

                    ensemble_outputs = torch.stack(ensemble_outputs)  # shape: [num_models, batch_size, num_classes]
                    averaged_logits = ensemble_outputs.mean(dim=0)  # shape: [batch_size, num_classes]
                    #only used for monitoring
                    loss_ensemble = criterion(averaged_logits, labels)
                    running_val_loss += loss_ensemble.item() * inputs.size(0)

                    ensemble_probs = torch.softmax(ensemble_outputs, dim=-1).mean(dim=0)  # shape: [batch_size, num_classes]
                    preds_ensemble = torch.argmax(ensemble_probs, dim=1)
                    num_correct_val += (preds_ensemble == labels).sum().item()
                    total_val_samples += labels.size(0)

            # Compute epoch-level validation metrics
            val_loss_ensemble = running_val_loss / total_val_samples
            val_accuracy_ensemble = 100.0 * num_correct_val / total_val_samples
            history['val_losses'].append(val_loss_ensemble)
            history['val_accuracies'].append(val_accuracy_ensemble)

            # --- EARLY STOPPING UPDATE ---
            if early_stopping:
                for m in range(self.num_models):
                    if not active_models[m]:
                        continue
                    # Compute average validation loss for model m.
                    val_loss_model = (individual_val_losses[m] / individual_val_counts[m]
                                      if individual_val_counts[m] > 0 else float('inf'))
                    # Check if there is improvement.
                    if best_val_losses[m] - val_loss_model > min_delta:
                        best_val_losses[m] = val_loss_model
                        patience_counters[m] = 0
                        best_model_states[m] = self.models[m].state_dict()
                    else:
                        patience_counters[m] += 1
                    # If patience is exceeded, mark this model as inactive.
                    if patience_counters[m] >= patience:
                        active_models[m] = False
                        self.models[m].load_state_dict(best_model_states[m])
                        print(f"Early stopping triggered for model {m} at epoch {epoch+1}.")

                # Break the epoch loop if all models have stopped.
                if not any(active_models):
                    break

            else:
                active_models = [True] * self.num_models # Reset active models for next epoch
        
        print(f"All models have stopped. Ending training at epoch {epoch+1}.")
        return history

    ##################################
    # FGSM
    ##################################

    def fgsm_attack(self, model, X, Y, criterion, epsilon):
        """
        Perform FGSM attack on the inputs.

        Args:
            model (nn.Module): The model to attack.
            X (torch.Tensor): Input tensor.
            Y (torch.Tensor): True labels.
            criterion (nn.Module): Loss function.
            epsilon (float): Perturbation magnitude.

        Returns:
            torch.Tensor: Perturbed inputs.
        """
        # requires_grad_() is an in-place operation
        #X.requires_grad = True
        X_adv = X.clone().detach().requires_grad_()
        outputs = model(X_adv)
        loss = criterion(outputs, Y)
        model.zero_grad()
        loss.backward()
        data_grad = X_adv.grad.data  # Compute gradient of loss w.r.t. input data
        sign_data_grad = data_grad.sign()
        perturbed_data = X_adv + epsilon * sign_data_grad  # Perturb based on sign of gradient
        perturbed_data = torch.clamp(perturbed_data, 0, 1).detach()
        return perturbed_data.detach_()

    ##################################
    # Prediction
    ##################################

    def predict(self, dataloader):
        predictions_list = []
        all_labels = []
        member_probs_list = []
        predictive_probs_list = []
        aleatoric = []
        aleatoric_median = []
        epistemic = []
        nll_vals = []
        
        self.eval()  # Set the ensemble to evaluation mode
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 1:
                    inputs = batch[0].to(self.device)
                    labels = None
                else:
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    
                if labels is not None:
                    all_labels.append(labels)

                ensemble_outputs = []
                for model in self.models:
                    outputs = model(inputs)  # Raw logits
                    ensemble_outputs.append(outputs)
                ensemble_outputs = torch.stack(ensemble_outputs)

                # Convert logits to probabilities
                batch_member_probs = F.softmax(ensemble_outputs, dim=-1)  # [num_models, batch_size, num_classes]
                member_probs_list.append(batch_member_probs.cpu())

                batch_ensemble_probs = torch.mean(batch_member_probs, dim=0)  # shape: [batch_size, num_classes]
                batch_predictive_probs, batch_predictions = torch.max(batch_ensemble_probs, dim=1)
                predictions_list.append(batch_predictions)
                predictive_probs_list.append(batch_predictive_probs)

                # Negative log-likelihood (NLL) computation
                if labels is not None:
                    nll = -torch.gather(torch.log(batch_ensemble_probs + 1e-12), dim=1, index=labels.unsqueeze(1)).squeeze()
                    nll_vals.append(nll)

                # Compute uncertainty measures
                batch_aleatoric_uncertainty = -torch.sum(batch_member_probs * torch.log(batch_member_probs.clamp(min=1e-12)), dim=-1).mean(dim=0)
                batch_aleatoric_median_uncertainty = -torch.sum(batch_member_probs * torch.log(batch_member_probs.clamp(min=1e-12)), dim=-1).median(dim=0).values
                aleatoric.append(batch_aleatoric_uncertainty)
                aleatoric_median.append(batch_aleatoric_median_uncertainty)
                batch_total_uncertainty = -torch.sum(batch_ensemble_probs * torch.log(batch_ensemble_probs.clamp(min=1e-12)), dim=-1)
                batch_epistemic_uncertainty = batch_total_uncertainty - batch_aleatoric_uncertainty
                epistemic.append(batch_epistemic_uncertainty)

            # Concatenate results along batch dimension.
            predictions = torch.cat(predictions_list, dim=0).cpu().numpy()
            predictive_probs = torch.cat(predictive_probs_list, dim=0).cpu().numpy()
            all_labels = torch.cat(all_labels, dim=0).cpu().numpy() if all_labels else None
            member_probs = torch.cat(member_probs_list, dim=1).cpu().numpy()
            aleatoric = torch.cat(aleatoric, dim=0).cpu().numpy()
            aleatoric_median = torch.cat(aleatoric_median, dim=0).cpu().numpy()
            epistemic = torch.cat(epistemic, dim=0).cpu().numpy()
            nll_vals = torch.cat(nll_vals, dim=0).cpu().numpy() if nll_vals else None

        return {
            "predictions": predictions,
            "probabilities": predictive_probs,         # shape: [num_samples]
            "labels": all_labels,
            "member_probabilities": member_probs,       # shape: [num_models, num_samples, num_classes]
            "aleatoric": aleatoric,
            "aleatoric_median": aleatoric_median,
            "epistemic": epistemic,
            "nll": nll_vals
        }
    




#--------------------------------------------------------------
# Deep ensemble for LSTM
#--------------------------------------------------------------




# new ensemble for dynamic and static split
class DeepEnsembleLSTM(nn.Module):
    def __init__(self, num_models, model_class, lr, weight_decay=0, **model_kwargs):
        """
        No changes in initialization except that models may now expect multi-inputs.
        """
        super(DeepEnsembleLSTM, self).__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([])
        for i in range(num_models):
            self.models.append(model_class(**model_kwargs))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                           for model in self.models]

    ##################################
    # Modified FGSM Attack to Support Multi-Input
    ##################################
    def fgsm_attack(self, model, inputs, Y, criterion, epsilon):
        """
        Modified FGSM attack that now accepts inputs as a tuple:
        (dynamic_inputs, static_inputs). The attack is applied to dynamic_inputs only.
        """
        if isinstance(inputs, (tuple, list)):
            dynamic_inputs, static_inputs = inputs
            # Clone dynamic_inputs and set gradient tracking.
            dynamic_inputs = dynamic_inputs.clone().detach().requires_grad_()
            outputs = model(dynamic_inputs, static_inputs)
            loss = criterion(outputs, Y)
            model.zero_grad()
            loss.backward()
            data_grad = dynamic_inputs.grad.data
            perturbed_dynamic = dynamic_inputs + epsilon * data_grad.sign()
            perturbed_dynamic = torch.clamp(perturbed_dynamic, 0, 1)
            # Return tuple with perturbed dynamic and original static inputs.
            return (perturbed_dynamic, static_inputs)
        else:
            # Fallback for single tensor inputs.
            X_adv = inputs.clone().detach().requires_grad_()
            outputs = model(X_adv)
            loss = criterion(outputs, Y)
            model.zero_grad()
            loss.backward()
            data_grad = X_adv.grad.data
            sign_data_grad = data_grad.sign()
            perturbed_data = X_adv + epsilon * sign_data_grad
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            return perturbed_data.detach_()

    ##################################
    # Modified Training Loop for Multi-Input
    ##################################
    def train_model(self, trainloader, validationloader, criterion, num_epochs, epsilon,
                     adversarial_training=True, 
                     early_stopping = True,
                     patience=3, min_delta=1e-6):
        """
        Updated to unpack (dynamic_inputs, static_inputs, labels) from the dataloader.
        Also, calls the model with two inputs.
        """
        history = {'train_losses': [], 'train_accuracies': [], 'val_losses': [], 'val_accuracies': []}
        best_val_losses = [float('inf')] * self.num_models
        patience_counters = [0] * self.num_models
        active_models = [True] * self.num_models
        best_model_states = [model.state_dict() for model in self.models]

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            num_correct_train = 0
            total_train_samples = 0

            # --- Training Loop (Changed Input Unpacking) ---
            for dynamic_inputs, static_inputs, labels in trainloader:
                dynamic_inputs = dynamic_inputs.to(self.device)
                static_inputs = static_inputs.to(self.device)
                labels = labels.to(self.device)
                ensemble_outputs = []

                for m, model in enumerate(self.models):
                    if not active_models[m]:
                        with torch.no_grad():
                            # Call the model with two inputs
                            outputs = model(dynamic_inputs, static_inputs)
                        ensemble_outputs.append(outputs)
                        continue

                    optimizer = self.optimizers[m]
                    optimizer.zero_grad()

                    if adversarial_training:
                        # Apply FGSM attack to the tuple of inputs.
                        perturbed_inputs = self.fgsm_attack(model, (dynamic_inputs, static_inputs), labels, criterion, epsilon)
                        # Concatenate the clean and adversarial inputs along the batch dimension.
                        combined_dynamic = torch.cat([dynamic_inputs, perturbed_inputs[0]], dim=0)
                        combined_static = torch.cat([static_inputs, perturbed_inputs[1]], dim=0)
                        combined_labels = torch.cat([labels, labels], dim=0)
                    else:
                        combined_dynamic = dynamic_inputs
                        combined_static = static_inputs
                        combined_labels = labels

                    # Call model with two inputs.
                    outputs = model(combined_dynamic, combined_static)
                    loss = criterion(outputs, combined_labels)
                    loss.backward()
                    optimizer.step()

                    # Only use outputs from clean inputs for aggregation.
                    ensemble_outputs.append(outputs[:len(labels)])

                ensemble_outputs = torch.stack(ensemble_outputs)  # shape: [num_models, batch_size, num_classes]
                averaged_logits = ensemble_outputs.mean(dim=0)  # shape: [batch_size, num_classes]
                loss_ensemble = criterion(averaged_logits, labels)
                running_loss += loss_ensemble.item() * labels.size(0)

                ensemble_probs = torch.softmax(ensemble_outputs, dim=-1).mean(dim=0)
                preds = torch.argmax(ensemble_probs, dim=1)
                num_correct_train += (preds == labels).sum().item()
                total_train_samples += labels.size(0)

            train_loss = running_loss / total_train_samples
            train_accuracy = 100.0 * num_correct_train / total_train_samples
            history['train_losses'].append(train_loss)
            history['train_accuracies'].append(train_accuracy)

            # --- Validation Loop (Changed Input Unpacking) ---
            self.eval()
            running_val_loss = 0.0
            num_correct_val = 0
            total_val_samples = 0
            individual_val_losses = [0.0] * self.num_models
            individual_val_counts = [0] * self.num_models

            with torch.no_grad():
                for dynamic_inputs, static_inputs, labels in validationloader:
                    dynamic_inputs = dynamic_inputs.to(self.device)
                    static_inputs = static_inputs.to(self.device)
                    labels = labels.to(self.device)
                    ensemble_outputs = []
                    for m, model in enumerate(self.models):
                        outputs = model(dynamic_inputs, static_inputs)
                        ensemble_outputs.append(outputs)
                        if active_models[m]:
                            loss_model = criterion(outputs, labels)
                            individual_val_losses[m] += loss_model.item() * dynamic_inputs.shape[0]
                            individual_val_counts[m] += dynamic_inputs.shape[0]
                    ensemble_outputs = torch.stack(ensemble_outputs)
                    averaged_logits = ensemble_outputs.mean(dim=0)
                    loss_ensemble = criterion(averaged_logits, labels)
                    running_val_loss += loss_ensemble.item() * labels.size(0)

                    ensemble_probs = torch.softmax(ensemble_outputs, dim=-1).mean(dim=0)
                    preds_ensemble = torch.argmax(ensemble_probs, dim=1)
                    num_correct_val += (preds_ensemble == labels).sum().item()
                    total_val_samples += labels.size(0)

            val_loss_ensemble = running_val_loss / total_val_samples
            val_accuracy_ensemble = 100.0 * num_correct_val / total_val_samples
            history['val_losses'].append(val_loss_ensemble)
            history['val_accuracies'].append(val_accuracy_ensemble)

            # --- Early Stopping Update (No change in logic, just using new validation losses) ---
            if early_stopping:
                for m in range(self.num_models):
                    if not active_models[m]:
                        continue
                    # Compute average validation loss for model m.
                    val_loss_model = (individual_val_losses[m] / individual_val_counts[m]
                                      if individual_val_counts[m] > 0 else float('inf'))
                    # Check if there is improvement.
                    if best_val_losses[m] - val_loss_model > min_delta:
                        best_val_losses[m] = val_loss_model
                        patience_counters[m] = 0
                        best_model_states[m] = self.models[m].state_dict()
                    else:
                        patience_counters[m] += 1
                    # If patience is exceeded, mark this model as inactive.
                    if patience_counters[m] >= patience:
                        active_models[m] = False
                        self.models[m].load_state_dict(best_model_states[m])
                        print(f"Early stopping triggered for model {m} at epoch {epoch+1}.")

                # Break the epoch loop if all models have stopped.
                if not any(active_models):
                    break
            else:
                active_models = [True] * self.num_models # Reset active models for next epoch
        
        print(f"All models have stopped. Ending training at epoch {epoch+1}.")
        return history

    ##################################
    # Modified Prediction Loop for Multi-Input
    ##################################
    def predict(self, dataloader):
        """
        Updated to expect batches with (dynamic_inputs, static_inputs, labels) or (dynamic_inputs, static_inputs).
        """
        predictions_list = []
        all_labels = []
        member_probs_list = []
        predictive_probs_list = []
        aleatoric = []
        aleatoric_median = []
        epistemic = []
        nll_vals = []

        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Unpack dynamic and static inputs (and optionally labels)
                if len(batch) == 3:
                    dynamic_inputs, static_inputs, labels = batch
                    dynamic_inputs = dynamic_inputs.to(self.device)
                    static_inputs = static_inputs.to(self.device)
                    labels = labels.to(self.device)
                    all_labels.append(labels)
                else:
                    dynamic_inputs, static_inputs = batch
                    dynamic_inputs = dynamic_inputs.to(self.device)
                    static_inputs = static_inputs.to(self.device)

                ensemble_outputs = []
                for model in self.models:
                    outputs = model(dynamic_inputs, static_inputs)
                    ensemble_outputs.append(outputs)
                ensemble_outputs = torch.stack(ensemble_outputs)

                batch_member_probs = F.softmax(ensemble_outputs, dim=-1)
                member_probs_list.append(batch_member_probs.cpu())

                batch_ensemble_probs = torch.mean(batch_member_probs, dim=0)
                batch_predictive_probs, batch_predictions = torch.max(batch_ensemble_probs, dim=1)
                predictions_list.append(batch_predictions)
                predictive_probs_list.append(batch_predictive_probs)

                if len(batch) == 3:
                    nll = -torch.gather(torch.log(batch_ensemble_probs + 1e-12), dim=1, index=labels.unsqueeze(1)).squeeze()
                    nll_vals.append(nll)
                    batch_aleatoric_uncertainty = -torch.sum(batch_member_probs * torch.log(batch_member_probs.clamp(min=1e-12)), dim=-1).mean(dim=0)
                    batch_aleatoric_median_uncertainty = -torch.sum(batch_member_probs * torch.log(batch_member_probs.clamp(min=1e-12)), dim=-1).median(dim=0).values
                    aleatoric.append(batch_aleatoric_uncertainty)
                    aleatoric_median.append(batch_aleatoric_median_uncertainty)
                    batch_total_uncertainty = -torch.sum(batch_ensemble_probs * torch.log(batch_ensemble_probs.clamp(min=1e-12)), dim=-1)
                    batch_epistemic_uncertainty = batch_total_uncertainty - batch_aleatoric_uncertainty
                    epistemic.append(batch_epistemic_uncertainty)

            predictions = torch.cat(predictions_list, dim=0).cpu().numpy()
            predictive_probs = torch.cat(predictive_probs_list, dim=0).cpu().numpy()
            all_labels = torch.cat(all_labels, dim=0).cpu().numpy() if all_labels else None
            member_probs = torch.cat(member_probs_list, dim=1).cpu().numpy()
            aleatoric = torch.cat(aleatoric, dim=0).cpu().numpy() if aleatoric else None
            aleatoric_median = torch.cat(aleatoric_median, dim=0).cpu().numpy() if aleatoric_median else None
            epistemic = torch.cat(epistemic, dim=0).cpu().numpy() if epistemic else None
            nll_vals = torch.cat(nll_vals, dim=0).cpu().numpy() if nll_vals else None

        return {
            "predictions": predictions,
            "probabilities": predictive_probs,
            "labels": all_labels,
            "member_probabilities": member_probs,
            "aleatoric": aleatoric,
            "aleatoric_median": aleatoric_median,
            "epistemic": epistemic,
            "nll": nll_vals
        }
