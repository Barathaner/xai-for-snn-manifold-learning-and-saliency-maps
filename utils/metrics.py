def print_batch_accuracy(predictions,targets):
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    print(f"Batch Accuracy: {accuracy:.4f}")
