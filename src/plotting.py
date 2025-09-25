import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score



#visualize the training and validation loss and accuracy
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs):
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()




def plot_confusion_matrix(model, test_loader, class_names, c1_prim = None, c2_prim = None, epi_scalar_prim = None):
    if c1_prim is not None and c2_prim is not None and epi_scalar_prim is not None:
        # If the model has a custom predict method, use it.
        results = model.predict(test_loader, c1_prim, c2_prim, epi_scalar_prim)
    else:
        results = model.predict(test_loader)
    predictions, labels  = results["predictions"], results["labels"]
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(xticks_rotation='vertical', ax=ax, cmap='Blues')
    plt.show()



def plot_ensemble_coverage_vs_metrics(member_probs, true_labels, resolution=150):

    # If member_probs is 2D (single model), expand it to 3D.
    if member_probs.ndim == 2:
        member_probs = np.expand_dims(member_probs, axis=0) #shape: [1, num_samples, num_classes]

    num_models = member_probs.shape[0]

    model_names = [f"Model {i+1}" for i in range(num_models)]
    results = []
    n_classes = member_probs.shape[-1]

    for i in range(num_models):
        probability_vector = member_probs[i]  # shape: [num_samples, num_classes]

        thresholds = np.append(0.0, (1 - np.logspace(np.log(0.955), -5, num=resolution))) # [0.0, 0.1, ..., 0.99]

        predicted_labels = probability_vector.argmax(axis=1)
        predicted_class_probs = probability_vector.max(axis=1)

        base_coverage = np.mean(predicted_labels == true_labels)
        base_set_size = 1.0
        base_mass = np.mean(predicted_class_probs)

        # Lists to store metrics across thresholds.
        coverage_list = [base_coverage]
        avg_set_size_list = [base_set_size]
        avg_mass_list = [base_mass]

        # Compute the Area Over the Curve (AOC) metric.
        area_over_curve = 0.0

        # Loop over thresholds (skipping the first baseline threshold).
        for thresh in thresholds[1:]:
            predictive_sets = []
            cumulative_masses = []
            for sample_prob in probability_vector:
                sorted_idx = np.argsort(sample_prob)[::-1]
                cum_prob = 0.0
                selected = []
                # Build predictive set by accumulating probabilities until threshold is reached.
                for cls in sorted_idx:
                    cum_prob += sample_prob[cls]
                    selected.append(cls)
                    if cum_prob >= thresh:
                        break
                predictive_sets.append(selected)
                cumulative_masses.append(cum_prob)

            # Calculate coverage: fraction of samples where the true label is in the predictive set.
            coverage = np.mean([true_labels[j] in predictive_sets[j]
                                for j in range(len(true_labels))])
            coverage_list.append(coverage)
            avg_set_size_list.append(np.mean([len(ps) for ps in predictive_sets]))
            avg_mass_list.append(np.mean(cumulative_masses))

            # Update AOC using a trapezoidal approximation.
            area_over_curve += (1.0 - 0.5 * (coverage_list[-1] + coverage_list[-2])) * \
                               (avg_set_size_list[-1] - avg_set_size_list[-2])

            # Stop early if full coverage is reached.
            if np.isclose(coverage, 1.0):
                break

        # Extend the AOC calculation to cover the full range of set sizes.
        area_over_curve += (1.0 - 0.5 * (1.0 + coverage_list[-1])) * (n_classes - avg_set_size_list[-1])
        area_over_curve /= (n_classes - 1)

        # Compute log(AOC) and update the model's label.
        log_aoc = np.log(area_over_curve)
        model_label = model_names[i] + f" (log(AOC)={log_aoc:.4f})"

        # Store results for the current model.
        results.append({
            "thresholds": thresholds[:len(coverage_list)],
            "coverage": coverage_list,
            "avg_set_size": avg_set_size_list,
            "avg_mass": avg_mass_list,
            "model_label": model_label,
        })

    # Now that all computations are done, set up the plots.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # Configure each subplot.
    axes[0, 0].set_title("Coverage vs. Predictive Set Size")
    axes[0, 0].set_xlabel("Average Predictive Set Size(Average # Classes looked at)")
    axes[0, 0].set_ylabel("Coverage(Fraction of right labels currently covered)")
    axes[0, 0].grid(True)

    axes[0, 1].set_title("Coverage vs. Threshold")
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Coverage")
    axes[0, 1].grid(True)

    axes[1, 0].set_title("Predictive Set Size vs. Threshold")
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("Average Predictive Set Size")
    axes[1, 0].grid(True)

    axes[1, 1].set_title("Coverage vs. Predictive Mass")
    axes[1, 1].set_xlabel("Average Predictive Mass(Average of probabilities inlcuded => max 1)")
    axes[1, 1].set_ylabel("Coverage")
    axes[1, 1].grid(True)

    # Plot the computed curves for each model.
    for res in results:
        axes[0, 0].plot(res["avg_set_size"], res["coverage"], label=res["model_label"])
        axes[0, 1].plot(res["thresholds"], res["coverage"], label=res["model_label"])
        axes[1, 0].plot(res["thresholds"], res["avg_set_size"], label=res["model_label"])
        axes[1, 1].plot(res["avg_mass"], res["coverage"], label=res["model_label"])

    # Add legends to all subplots.
    for ax in axes.flat:
        ax.legend()

    # Add a diagonal reference line to the Coverage vs. Predictive Mass plot.
    xlim = axes[1, 1].get_xlim()
    ylim = axes[1, 1].get_ylim()
    axes[1, 1].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[1, 1].set_xlim(xlim)
    axes[1, 1].set_ylim(ylim)

    plt.show()




def plot_accuracy_rejection_curves(member_probs, true_labels, rejection_points=100, plot = False, title_name = None):
    """
    This function supports two input types for the prediction probabilities:
      - A 3D numpy array of shape [num_models, num_samples, num_classes] (ensemble predictions)
      - A 2D numpy array of shape [num_samples, num_classes] (single model predictions)

    Returns:
        list: A list of AUARC values, one for each model.
    """
    # If input is 2D (single model), convert it to 3D with a singleton model dimension.
    if member_probs.ndim == 2:
        member_probs = np.expand_dims(member_probs, axis=0)

    num_models = member_probs.shape[0]
    num_samples = member_probs.shape[1]

    rejection_thresholds = np.linspace(0.0, 0.99, rejection_points)
    accepted_counts = (num_samples * (1 - rejection_thresholds)).astype(int) # number of samples picked N*[0.99,0.98,...]

    arc_auc_list = []
    model_accuracies = []

    # Loop over each model in the ensemble.
    for i in range(num_models):
        model_probs = member_probs[i]  # shape: [num_samples, num_classes]
        confidences = np.max(model_probs, axis=-1)

        sorted_indices = np.argsort(confidences)[::-1]  # descending order [0.9, 0.8, 0.7, ...] but args
        sorted_true = true_labels[sorted_indices]

        accuracies = []
        total_accuracy = 0.0

        # Always order by confidence, but starting with many samples and decreasing
        for count in accepted_counts:
            #print(count)
            if count > 0:
                accepted_true = sorted_true[:count]
                accepted_predicted = np.argmax(model_probs[sorted_indices[:count]], axis=-1)
                accuracy = np.mean(accepted_predicted == accepted_true) # averageing [1,0,1,...]
            else:
                accuracy = 0.0
            accuracies.append(accuracy)
            total_accuracy += accuracy

        # Average accuracy over all thresholds (AUARC)
        avg_arc = total_accuracy / len(rejection_thresholds)
        arc_auc_list.append(avg_arc)
        model_accuracies.append(accuracies)

    accuracies = np.stack(model_accuracies, axis=0) # shape: [num_models, num_thresholds]

    if plot:
        plt.figure(figsize=(10, 6))
        for i in range(num_models):
            plt.plot(rejection_thresholds, accuracies[i], label=f"Model {i + 1}")
        plt.xlabel("Rejection Rate")
        plt.ylabel("Accuracy")
        if title_name:
            plt.title(f"{title_name}_Accuracy-Rejection Curves")
        else:
            plt.title("Accuracy-Rejection Curves")
        plt.legend()
        plt.show()

    return {
        "AUARC": arc_auc_list,
        "accuracies": accuracies
        }



#-----------------------------------------------------------------------------
### Metrics in addition to NLL
#-----------------------------------------------------------------------------

def get_coverage_threshold_and_size(probs, true_labels, 
                                    target_coverage=0.90, resolution=150):
    """
    For a single model’s probability matrix `probs` ([n_samples, n_classes]),
    find the minimal cumulative‐mass threshold at which you cover the true 
    label in at least `target_coverage` of the cases, and return that 
    threshold along with the corresponding average predictive‐set size.
    """
    n_samples, n_classes = probs.shape

    # threshold schedule
    thresholds = np.append(
        0.0,
        1.0 - np.logspace(np.log(0.955), -5, num=resolution)
    )
    # pre‐rank classes by probability per sample
    ranked = np.argsort(probs, axis=1)[:, ::-1]  # [n_samples, n_classes]

    coverages = []
    set_sizes = []

    for t in thresholds:
        hit = 0
        sizes = []
        for i in range(n_samples):
            cum = 0.0
            sz = 0
            for cls in ranked[i]:
                cum += probs[i, cls]
                sz += 1
                if cum >= t:
                    break
            sizes.append(sz)
            if true_labels[i] in ranked[i][:sz]:
                hit += 1

        coverage = hit / n_samples
        coverages.append(coverage)
        set_sizes.append(np.mean(sizes))

        if coverage >= 1.0:
            break

    cov_arr = np.array(coverages)
    idx = np.argmax(cov_arr >= target_coverage)
    #valid = np.where(cov_arr >= target_coverage)[0]
    #idx   = valid[-1]   # last index where coverage is still ≥ target
    return thresholds[idx], set_sizes[idx]


def get_auc(probs, true_labels):
    """
    Compute AUC for binary or multiclass classification.
    
    Args:
        probs        : np.ndarray, shape = (n_samples, n_classes)
                       Model's predicted probabilities for each class.
        true_labels  : np.ndarray, shape = (n_samples,)
                       Integer class labels in [0 .. n_classes-1].
    
    Returns:
        auc_score    : float
    """
    n_samples, n_classes = probs.shape

    if n_classes == 2:
        # Binary case: roc_auc_score expects a 1D array of "positive" scores
        # We assume class "1" is the positive class:
        positive_probs = probs[:, 1]
        return roc_auc_score(true_labels, positive_probs)

    else:
        # Multiclass: pass the full (n_samples, n_classes) array
        # and ask for one‐vs‐rest macro‐averaging
        return roc_auc_score(
            true_labels,
            probs,
            multi_class='ovr',
            average='macro'
        )


def compute_aurac(y_true, y_pred_proba, n_points=101):
    """
    Compute AURAC (Area Under Accuracy-Rejection Curve).
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels (ints).
    y_pred_proba : array-like of shape (n_samples, n_classes)
        Predicted probabilities (e.g. softmax outputs).
    n_points : int, default=101
        Number of rejection thresholds to evaluate.
    
    Returns
    -------
    aurac : float
        Area under the accuracy–rejection curve.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Predicted labels and confidences
    confidences = np.max(y_pred_proba, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Sort samples by confidence (descending)
    sorted_idx = np.argsort(confidences)[::-1]
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    # Rejection rates
    rejection_rates = np.linspace(0, 1, n_points)
    accuracies = []

    for r in rejection_rates:
        keep = int((1 - r) * len(y_true))
        if keep > 0:
            acc = np.mean(y_true_sorted[:keep] == y_pred_sorted[:keep])
        else:
            acc = 0.0
        accuracies.append(acc)

    # Approximate integral by averaging
    aurac = np.mean(accuracies)
    return aurac