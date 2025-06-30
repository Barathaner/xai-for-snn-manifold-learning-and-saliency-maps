def train_one_epoch(net, dataloader, optimizer, loss_fn, device):
    net.train()
    total_loss = 0
    correct = 0
    total = 0

    for events, labels in dataloader:
        events = events.to(device).float()
        labels = labels.to(device)

        batch_size = events.shape[0]

        for i in range(batch_size):
            sample = events[i].squeeze(0)  # [num_steps, 700]

            optimizer.zero_grad()

            spk_rec, mem_rec = net(sample)

            # Rate Coding: Summe aller Spikes als Logits
            spike_sums = spk_rec.sum(dim=0)

            loss = loss_fn(spike_sums, labels[i].unsqueeze(0))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            pred = torch.argmax(spike_sums).item()
            if pred == labels[i].item():
                correct += 1
            total += 1

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


num_epochs = 3
for epoch in range(1, num_epochs+1):
    print(f"\nEpoch {epoch}")
    train_one_epoch(ffn, train_dataloader, optimizer, loss, device)
