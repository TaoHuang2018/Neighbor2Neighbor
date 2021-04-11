1. about Synthetic Data Generator

2. about Random Image Sub-sampler

3. about Training Strategy

```
for iteration, clean in enumerate(Dataloader):
  # preparing synthetic noisy images
  clean = clean / 255.0
  clean = clean.cuda()
  noisy = noise_adder.add_train_noise(clean)
  optimizer.zero_grad()
  # generating a sub-image pair
  mask1, mask2 = generate_mask_pair(noisy)
  noisy_sub1 = generate_subimages(noisy, mask1)
  noisy_sub2 = generate_subimages(noisy, mask2)
  # preparing for the regularization term
  with torch.no_grad():
    noisy_denoised = network(noisy)
  noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
  noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)
  # calculating the loss 
  noisy_output = network(noisy_sub1)
  noisy_target = noisy_sub2
  Lambda = epoch / n_epoch * ratio
  diff = noisy_output - noisy_target
  exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
  loss1 = torch.mean(diff**2)
  loss2 = Lambda * torch.mean((diff - exp_diff)**2)
  loss_all = loss1 + loss2
  loss_all.backward()
  optimizer.step()
```

