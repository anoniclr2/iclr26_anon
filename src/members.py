import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class SimplerSimpleNN(nn.Module):
    """
    Smaller SimpleNN matching the architecture from Calibration notebook.
    Architecture: input -> 128 -> 64 -> output with dropout(0.25)
    """
    def __init__(self, input_shape=2, output_shape=2, dropout_rate=0.25):
        super(SimplerSimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_shape)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def train_model(self, train_loader, val_loader,
                    criterion=nn.CrossEntropyLoss(),
                    optimizer=None, num_epochs=10,
                    early_stopping=True,
                    patience=2, min_delta=1e-6):

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self.train()  # set mode to training
            train_counts = 0
            correct_preds = 0
            running_loss = 0.0

            # Training loop:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)  # logits: shape [batch_size, num_classes]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                softmax_outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(softmax_outputs, 1)
                train_counts += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

            train_losses.append(running_loss / train_counts)
            train_accuracies.append((correct_preds / train_counts) * 100)

            # Validation loop:
            self.eval()  # set mode to evaluation
            val_running_loss = 0.0
            val_counts = 0
            val_total_preds = 0
            val_correct_preds = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    val_counts += inputs.size(0)

                    softmax_outputs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(softmax_outputs, 1)
                    val_total_preds += labels.size(0)
                    val_correct_preds += (predicted == labels).sum().item()

            val_loss = val_running_loss / val_counts
            val_losses.append(val_loss)
            val_accuracies.append((val_correct_preds / val_total_preds) * 100)

            # Early stoppage
            if early_stopping:
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.load_state_dict(best_model_state)
                    break
        if early_stopping:
            print(f"Early stopping triggered at epoch {epoch+1}.")
        else:
            print(f"Training completed after {num_epochs} epochs.")
            
        return train_losses, train_accuracies, val_losses, val_accuracies
    
    def predict(self, test_loader):
        """
        Returns predictions and labels for all samples in test_loader.
        
        If labels are provided in the loader, it also computes the negative log-likelihood (NLL)
        for the true class.
        """
        self.eval()
        predictions_list = []
        labels_list = []
        max_probs_list = []
        probs_list = []
    
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 1:
                    inputs = batch[0].to(self.device)
                    labels = None
                else:
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)

                outputs = self(inputs)
                probabilities = F.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probabilities, 1)
            
                probs_list.append(probabilities)
                predictions_list.append(predicted)
                max_probs_list.append(max_probs)
                labels_list.append(labels)
    
        # Concatenate results along the batch dimension
        predictions_cat = torch.cat(predictions_list, dim=0).cpu().numpy()
        max_probs_cat = torch.cat(max_probs_list, dim=0).cpu().numpy()
        probs_cat = torch.cat(probs_list, dim=0)
    
        # When labels are not provided in the test_loader
        if labels_list[0] is None:
            return {
            "predictions": predictions_cat,
            "probabilities": max_probs_cat,
            "probabilities_vect": probs_cat.cpu().numpy()
            }
        else:
            labels_cat = torch.cat(labels_list, dim=0)
        
            # Calculate NLL using the true class probabilities
            true_class_probs = torch.gather(probs_cat, 1, labels_cat.unsqueeze(1)).squeeze(1)
            nll = -torch.log(true_class_probs + 1e-12)      # shape: [num_samples]
        
        return {
            "predictions": predictions_cat,
            "probabilities": max_probs_cat,  # shape: [num_samples]
            "labels": labels_cat.cpu().numpy(),
            "probabilities_vect": probs_cat.cpu().numpy(),  # shape: [num_samples, num_classes]
            "nll": nll.cpu().numpy()
        }


class SimpleNN(nn.Module):
    def __init__(self, input_shape = 2, output_shape = 2):      # using cross entropy and logits => dim = num_classes
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def train_model(self, train_loader, val_loader,
                    criterion=nn.CrossEntropyLoss(),
                    optimizer=None, num_epochs=10,
                    early_stopping=True,
                    patience=2, min_delta=1e-6):

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self.train()  # set mode to training
            train_counts = 0
            correct_preds = 0
            running_loss = 0.0

            # Training loop:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                # Use self(inputs) instead of self.forward(inputs) for consistency:
                outputs = self(inputs)  # logits: shape [batch_size, num_classes]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)


                # Although softmax is not required for argmax, it’s fine for clarity:
                softmax_outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(softmax_outputs, 1)
                train_counts += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

            train_losses.append(running_loss / train_counts)
            train_accuracies.append((correct_preds / train_counts) * 100)

            # Validation loop:
            self.eval()  # set mode to evaluation
            val_running_loss = 0.0
            val_counts = 0
            val_total_preds = 0
            val_correct_preds = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    val_counts += inputs.size(0)

                    softmax_outputs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(softmax_outputs, 1)
                    val_total_preds += labels.size(0)
                    val_correct_preds += (predicted == labels).sum().item()

            val_loss = val_running_loss / val_counts
            val_losses.append(val_loss)
            val_accuracies.append((val_correct_preds / val_total_preds) * 100)

            # Optionally, print epoch summary:
            #print(f"Epoch {epoch+1}/{num_epochs}")
            #print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%")
            #print(f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%")

            # Early stoppage
            if early_stopping:
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.load_state_dict(best_model_state)
                    break
        if early_stopping:
            print(f"Early stopping triggered at epoch {epoch+1}.")
        else:
            print(f"Training completed after {num_epochs} epochs.")
            
        return train_losses, train_accuracies, val_losses, val_accuracies
    
    def predict(self, test_loader):
        """
        Returns predictions and labels for all samples in test_loader.
        
        If labels are provided in the loader, it also computes the negative log-likelihood (NLL)
        for the true class.
        """
        self.eval()
        predictions_list = []
        labels_list = []
        max_probs_list = []
        probs_list = []
    
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 1:
                    inputs = batch[0].to(self.device)
                    labels = None
                else:
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)

                outputs = self(inputs)
                probabilities = F.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probabilities, 1)
            
                probs_list.append(probabilities)
                predictions_list.append(predicted)
                max_probs_list.append(max_probs)
                labels_list.append(labels)
    
        # Concatenate results along the batch dimension
        predictions_cat = torch.cat(predictions_list, dim=0).cpu().numpy()
        max_probs_cat = torch.cat(max_probs_list, dim=0).cpu().numpy()
        probs_cat = torch.cat(probs_list, dim=0)
    
        # When labels are not provided in the test_loader
        if labels_list[0] is None:
            return {
            "predictions": predictions_cat,
            "probabilities": max_probs_cat,
            "probabilities_vect": probs_cat.cpu().numpy()
            }
        else:
            labels_cat = torch.cat(labels_list, dim=0)
        
            # Calculate NLL using the true class probabilities
            true_class_probs = torch.gather(probs_cat, 1, labels_cat.unsqueeze(1)).squeeze(1)
            nll = -torch.log(true_class_probs + 1e-12)      # shape: [num_samples]
        
        return {
            "predictions": predictions_cat,
            "probabilities": max_probs_cat,  # shape: [num_samples]
            "labels": labels_cat.cpu().numpy(),
            "probabilities_vect": probs_cat.cpu().numpy(),  # shape: [num_samples, num_classes]
            "nll": nll.cpu().numpy()
        }


#-------------------------------------------------------------------------------------------
# CNN model for MNIST dataset
#-------------------------------------------------------------------------------------------

class CnnNet(nn.Module):
    def __init__(self, output_shape=10, input_shape=28*28):
        super(CnnNet, self).__init__()
        # Set device (using "cuda" if available, similar to SimpleNN)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # For MNIST, assume single-channel 28x28 images.
        # Convolutional layers:
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)   # (1,28,28) -> (16,24,24)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)                             # (16,24,24) -> (16,12,12)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5)    # (16,12,12) -> (24,8,8)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)                             # (24,8,8) -> (24,4,4)
        
        # Fully connected layers:
        self.fc1 = nn.Linear(24 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, output_shape)
        
        # Move model to device immediately
        self.to(self.device)
    
    def forward(self, x):
        # Convolutional part:
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        # Flatten feature maps:
        x = x.view(x.size(0), -1)
        # Fully connected part:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, train_loader, val_loader,
                    criterion=nn.CrossEntropyLoss(),
                    optimizer=None, num_epochs=10,
                    early_stopping=True,
                    patience=3, min_delta=1e-6):
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.train()  # set mode to training
            running_loss = 0.0
            total_preds = 0
            correct_preds = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Accumulate loss weighted by batch size:
                running_loss += loss.item() * inputs.size(0)
                
                # Compute predictions and accuracy:
                softmax_outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(softmax_outputs, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()
            
            train_losses.append(running_loss / total_preds)
            train_accuracies.append((correct_preds / total_preds) * 100)
            
            # Validation loop:
            self.eval()  # set mode to evaluation
            val_running_loss = 0.0
            val_total = 0
            val_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    val_total += inputs.size(0)
                    
                    softmax_outputs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(softmax_outputs, 1)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_running_loss / val_total
            val_losses.append(val_loss)
            val_accuracies.append((val_correct / val_total) * 100)
            
            # Early stopping check:
            if early_stopping:
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.load_state_dict(best_model_state)
                    break
        if early_stopping:
            print(f"Early stopping triggered at epoch {epoch+1}.")
        else:
            print(f"Training completed after {num_epochs} epochs.")

        return train_losses, train_accuracies, val_losses, val_accuracies

    def predict(self, test_loader):
        """
        Returns predictions and labels for all samples in test_loader.
        If labels are available, also returns the negative log-likelihood (NLL)
        for the true class.
        """
        self.eval()
        predictions_list = []
        labels_list = []
        max_probs_list = []
        probs_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Check if the batch includes labels
                if len(batch) == 1:
                    inputs = batch[0].to(self.device)
                    labels = None
                else:
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                
                outputs = self(inputs)
                probabilities = F.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probabilities, 1)
                
                predictions_list.append(predicted)
                max_probs_list.append(max_probs)
                probs_list.append(probabilities)
                labels_list.append(labels)
        
        predictions_cat = torch.cat(predictions_list, dim=0).cpu().numpy()
        max_probs_cat = torch.cat(max_probs_list, dim=0).cpu().numpy()
        probs_cat = torch.cat(probs_list, dim=0)
        
        # When labels are not provided in the test loader:
        if labels_list[0] is None:
            return {
                "predictions": predictions_cat,
                "probabilities": max_probs_cat,
                "probabilities_vect": probs_cat.cpu().numpy()
            }
        else:
            labels_cat = torch.cat(labels_list, dim=0)
            true_class_probs = torch.gather(probs_cat, 1, labels_cat.unsqueeze(1)).squeeze(1)
            nll = -torch.log(true_class_probs + 1e-12)
            return {
                "predictions": predictions_cat,
                "probabilities": max_probs_cat,
                "labels": labels_cat.cpu().numpy(),
                "probabilities_vect": probs_cat.cpu().numpy(),
                "nll": nll.cpu().numpy()
            }



#-------------------------------------------------------------------------------------------
# CNN model for CIFAR-10 dataset
#-------------------------------------------------------------------------------------------


class Git_CNN(nn.Module):
    def __init__(self):
        super(Git_CNN, self).__init__()
        # Convolutional layers and batch norm layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.b2 = nn.BatchNorm2d(64)
        self.b3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout and fully-connected layers
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)
        
        # Set device and move model to the appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        # Forward pass through the conv layers with pooling and relu activations
        x = self.pool(F.relu(self.b1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.b2(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.b3(self.conv5(x))))
        # Flatten: assuming input images are sized such that after pooling, the feature map is 1x1 (e.g., CIFAR-10: 32x32)
        x = x.view(-1, 256)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.out(x)
        return x

    def train_model(self, train_loader, val_loader,
                    criterion=nn.CrossEntropyLoss(),
                    optimizer=None, num_epochs=10,
                    early_stopping = True,
                    patience=2, min_delta=1e-6):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self.train()  # Set model to training mode
            train_counts = 0
            correct_preds = 0
            running_loss = 0.0

            # Training loop:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                softmax_outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(softmax_outputs, 1)
                train_counts += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

            train_losses.append(running_loss / train_counts)
            train_accuracies.append((correct_preds / train_counts) * 100)

            # Validation loop:
            self.eval()  # Set model to evaluation mode
            val_running_loss = 0.0
            val_counts = 0
            val_correct_preds = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    val_counts += inputs.size(0)
                    softmax_outputs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(softmax_outputs, 1)
                    val_correct_preds += (predicted == labels).sum().item()

            val_loss = val_running_loss / val_counts
            val_losses.append(val_loss)
            val_accuracies.append((val_correct_preds / val_counts) * 100)

            # Early stopping:
            if early_stopping:
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.load_state_dict(best_model_state)
                    break
        if early_stopping:
            print(f"Early stopping triggered at epoch {epoch+1}.")
        else:
            print(f"Training completed after {num_epochs} epochs.")
        return train_losses, train_accuracies, val_losses, val_accuracies

    def predict(self, test_loader):
        """
        Returns predictions and labels (if provided) for all samples in test_loader.
        Also computes the negative log-likelihood (NLL) for the true class if labels are available.
        """
        self.eval()
        predictions_list = []
        labels_list = []
        max_probs_list = []
        probs_list = []

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 1:
                    inputs = batch[0].to(self.device)
                    labels = None
                else:
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)

                outputs = self(inputs)
                probabilities = F.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probabilities, 1)
                probs_list.append(probabilities)
                predictions_list.append(predicted)
                max_probs_list.append(max_probs)
                labels_list.append(labels)

        predictions_cat = torch.cat(predictions_list, dim=0).cpu().numpy()
        max_probs_cat = torch.cat(max_probs_list, dim=0).cpu().numpy()
        probs_cat = torch.cat(probs_list, dim=0)

        # When labels are not provided:
        if labels_list[0] is None:
            return {
                "predictions": predictions_cat,
                "probabilities": max_probs_cat,
                "probabilities_vect": probs_cat.cpu().numpy()
            }
        else:
            labels_cat = torch.cat(labels_list, dim=0)
            # Calculate NLL using the true class probabilities
            true_class_probs = torch.gather(probs_cat, 1, labels_cat.unsqueeze(1)).squeeze(1)
            nll = -torch.log(true_class_probs + 1e-12)
            return {
                "predictions": predictions_cat,
                "probabilities": max_probs_cat,
                "labels": labels_cat.cpu().numpy(),
                "probabilities_vect": probs_cat.cpu().numpy(),
                "nll": nll.cpu().numpy()
            }



#-------------------------------------------------------------------------------------------
# LSTM for stock prediction
#-------------------------------------------------------------------------------------------


class StockLSTM(nn.Module):
    def __init__(self, dynamic_input_shape, static_input_shape, hidden_shape, num_layers, output_shape, dropout=0.0):
        super(StockLSTM, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dynamic_input_shape = dynamic_input_shape
        self.static_input_shape = static_input_shape
        self.hidden_shape = hidden_shape
        self.num_layers = num_layers
        self.output_shape = output_shape

        # LSTM for dynamic features.
        self.lstm = nn.LSTM(dynamic_input_shape, hidden_shape, num_layers, batch_first=True, dropout=dropout)
        # Fully connected layer takes concatenated [LSTM output, static features].
        self.fc = nn.Linear(hidden_shape + static_input_shape, output_shape)

        self.to(self.device)

    def forward(self, dynamic_x, static_x):
        """
        Parameters:
            dynamic_x: Tensor of shape (batch_size, seq_length, dynamic_input_size)
            static_x: Tensor of shape (batch_size, static_input_size)
        Returns:
            logits: Tensor of shape (batch_size, output_size)
        """
        self.lstm.flatten_parameters() # This is necessary for performance.
        out, _ = self.lstm(dynamic_x)  # out: (batch_size, seq_length, hidden_size)
        last_output = out[:, -1, :]     # take the output at the last time step
        combined = torch.cat([last_output, static_x], dim=1)  # shape: (batch_size, hidden_size + static_input_size)
        logits = self.fc(combined)
        return logits

    def train_model(self, train_loader, val_loader, criterion=nn.CrossEntropyLoss(),
                    optimizer=None, num_epochs=10,
                    early_stopping=True,
                    patience=2, min_delta=1e-6):
        """
        Trains the model using provided DataLoaders.
        Assumes each batch is a tuple (dynamic_inputs, static_inputs, labels).
        """
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            total_preds = 0
            correct_preds = 0
            #more data to unpack now
            for dynamic_inputs, static_inputs, labels in train_loader:
                dynamic_inputs = dynamic_inputs.to(self.device)
                static_inputs = static_inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(dynamic_inputs, static_inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * dynamic_inputs.size(0)
                _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

            epoch_loss = running_loss / total_preds
            epoch_acc = 100 * correct_preds / total_preds
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            # Validation loop
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for dynamic_inputs, static_inputs, labels in val_loader:
                    dynamic_inputs = dynamic_inputs.to(self.device)
                    static_inputs = static_inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.forward(dynamic_inputs, static_inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * dynamic_inputs.size(0)
                    _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_epoch_loss = val_loss / val_total
            val_epoch_acc = 100 * val_correct / val_total
            val_losses.append(val_epoch_loss)
            val_accuracies.append(val_epoch_acc)

            #print(f"Epoch {epoch+1}/{num_epochs}:")
            #print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
            #print(f"  Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%\n")

            scheduler.step()

            # Early stopping logic
            if early_stopping:
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.load_state_dict(best_model_state)
                    break
        if early_stopping:
            print(f"Early stopping triggered at epoch {epoch+1}.")
        else:
            print(f"Training completed after {num_epochs} epochs.")

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, loader):
        """
        Runs inference on data from loader.
        Assumes each batch is a tuple (dynamic_inputs, static_inputs, labels).
        Returns a dictionary with keys: predictions, labels, probabilities.
        """
        self.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_nll = []
        all_logits = []

        with torch.no_grad():
            for dynamic_inputs, static_inputs, labels in loader:
                dynamic_inputs = dynamic_inputs.to(self.device)
                static_inputs = static_inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(dynamic_inputs, static_inputs)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
                all_logits.append(outputs.cpu())

                # Compute NLL for this batch
                true_class_probs = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)
                nll_batch = -torch.log(true_class_probs + 1e-12)
                all_nll.append(nll_batch.cpu())

        predictions = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        probabilities = torch.cat(all_probs).numpy()
        nll = torch.cat(all_nll).numpy()
        logits = torch.cat(all_logits).numpy()


        return {"predictions": predictions,
                "labels": labels,
                "probabilities": probabilities,
                "nll": nll,
                "logits": logits}

    


