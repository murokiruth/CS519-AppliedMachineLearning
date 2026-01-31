import numpy as np

# step 2c Compute weighted error rate 
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
yhat = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, -1])
correct = (y == yhat)
weights = np.array([0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.167, 0.167, 0.167, 0.072])
epsilon = np.sum(weights[~correct])
print(f'weighted error rate(epsilon) is {epsilon:.3f}')

# step 2d Compute coefficient
alpha_j = 0.5 * np.log((1 - epsilon) / epsilon)
print(f'coefficient is(alpha_j) {alpha_j:.3f}')

#step 2e Update weights
update_if_correct = weights * np.exp(-alpha_j * 1 * 1)
print(update_if_correct)

update_if_wrong = weights * np.exp(-alpha_j * 1 * -1)
print(update_if_wrong)

updated_weights = np.where(correct == 1,
                   update_if_correct,
                   update_if_wrong
                   )

print(f'Updated weights are {[f'{w:.3f}' for w in updated_weights]}')

#step 2f Normalize weights to sum to 1
normalized_weights = updated_weights / np.sum(updated_weights)
print(f'Normalized weights to sum to 1 are {[f'{w:.3f}' for w in normalized_weights]}')
