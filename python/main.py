import numpy as np

np.random.seed(69)

SAMPLES = 10
EPOCHS = 100
LOG_FREQ = 10
LR = 1e-1
def main():
    X = np.random.rand(SAMPLES, 1, 1)
    Y = X * 0.1 + 1
    w1 = np.random.rand(2, 1)
    b1 = np.random.rand(2, 1)

    w2 = np.random.rand(2, 2)
    b2 = np.random.rand(2, 1)

    w3 = np.random.rand(1, 2)
    b3 = np.random.rand(1, 1)


    def activation(input):
        return input * -1


    def activation_d(input):
        return np.zeros_like(input) - 1


    for epoch in range(EPOCHS):
        total_loss = 0
        for sample in range(SAMPLES):
            x = X[sample]
            y = Y[sample]
            a1 = np.matmul(w1, x) + b1
            a2 = np.matmul(w2, activation(a1)) + b2
            a3 = np.matmul(w3, activation(a2)) + b3

            loss = 0.5 * (y - activation(a3)) ** 2

            bd3 = (y - activation(a3)) * activation_d(a3)
            wd3 = np.matmul(bd3, a2.T)

            bd2 = np.matmul(w3.T, bd3) * activation_d(a2)
            wd2 = np.matmul(bd2, a1.T)

            bd1 = np.matmul(w2.T, bd2) * activation_d(a1)
            wd1 = np.matmul(bd1, x.T)

            w1 += wd1 * LR
            b1 += bd1 * LR

            w2 += wd2 * LR
            b2 += bd2 * LR

            w3 += wd3 * LR
            b3 += bd3 * LR

            total_loss += loss
        if epoch % LOG_FREQ == 0:
            print(f"Epoch: {epoch}, Loss: {total_loss}")
if __name__ == "__main__":
    main()
