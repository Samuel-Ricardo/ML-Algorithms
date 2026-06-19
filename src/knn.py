#%%
import numpy as np
from collections import Counter
#%%

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _distance(self, x1, x2):
        return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))

    def _predict(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        return [self._predict(x) for x in np.array(X)]
#%%

X_train = [[1, 2], [2, 3], [3, 3], [6, 5], [7, 7], [8, 6]]
y_train = ['A', 'A', 'A', 'B', 'B', 'B']
X_test = [[5, 5]]

model = KNN(k=3)
model.fit(X_train, y_train)
print(f"Predição para {X_test}: {model.predict(X_test)}")
#%%

print("\n" + "=" * 50)
print("TESTES COM DIFERENTES VALORES DE k")
print("=" * 50)

X_test_multiple = [[5, 5], [2, 2], [7, 6], [4, 4], [1, 1]]

for k in [1, 3, 5]:
    model_k = KNN(k=k)
    model_k.fit(X_train, y_train)
    preds = model_k.predict(X_test_multiple)
    print(f"\nk={k}:")
    for point, pred in zip(X_test_multiple, preds):
        print(f"  Ponto {point} -> Classe '{pred}'")
#%%

print("\n" + "=" * 50)
print("VALIDAÇÃO COM DATASET MAIOR")
print("=" * 50)

np.random.seed(42)
cluster_A = np.random.randn(20, 2) + np.array([0, 0])
cluster_B = np.random.randn(20, 2) + np.array([5, 5])
cluster_C = np.random.randn(20, 2) + np.array([10, 0])

X_full = np.vstack([cluster_A, cluster_B, cluster_C])
y_full = ['A'] * 20 + ['B'] * 20 + ['C'] * 20
indices = np.arange(len(X_full))
np.random.shuffle(indices)

split = int(0.8 * len(X_full))
train_idx = indices[:split]
test_idx = indices[split:]

X_tr = X_full[train_idx]
y_tr = [y_full[i] for i in train_idx]
X_te = X_full[test_idx]
y_te = [y_full[i] for i in test_idx]

model_val = KNN(k=3)
model_val.fit(X_tr, y_tr)
preds_val = model_val.predict(X_te)

correct = sum(1 for pred, real in zip(preds_val, y_te) if pred == real)
accuracy = correct / len(y_te)
print(f"\nAcurácia com k=3: {accuracy * 100:.1f}% ({correct}/{len(y_te)})")

print("\nPrimeiras 5 predições:")
print(f"{'Ponto':<25} {'Real':<8} {'Previsto':<10} {'Status'}")
print("-" * 55)
for i in range(min(5, len(y_te))):
    status = "OK" if preds_val[i] == y_te[i] else "ERRO"
    print(f"{str(X_te[i].round(2)):<25} {y_te[i]:<8} {preds_val[i]:<10} {status}")
#%%

print("\n" + "=" * 50)
print("IMPACTO DE k NA ACURÁCIA")
print("=" * 50)

for k in [1, 2, 3, 5, 7, 9]:
    model_analysis = KNN(k=k)
    model_analysis.fit(X_tr, y_tr)
    preds_k = model_analysis.predict(X_te)
    acc = sum(1 for pred, real in zip(preds_k, y_te) if pred == real) / len(y_te)
    print(f"  k={k:<3} -> Acurácia: {acc * 100:.1f}%")
#%%
