# k-of-n gate lookup (true-positive sensitivity s = 0.9)

## p_fp = 0.01, FAR < 0.001

| ρ | n = 3 | n = 5 | n = 10 |
|---|---|---|---|
| 0.0 | k=2 (TAR 0.97) | k=2 (TAR 1.00) | k=3 (TAR 1.00) |
| 0.2 | k=3 (TAR 0.77) | k=4 (TAR 0.87) | k=7 (TAR 0.91) |
| 0.5 | none (min FAR 3.4e-03) | none (min FAR 2.0e-03) | none (min FAR 1.0e-03) |

## p_fp = 0.01, FAR < 0.0001

| ρ | n = 3 | n = 5 | n = 10 |
|---|---|---|---|
| 0.0 | k=3 (TAR 0.73) | k=3 (TAR 0.99) | k=4 (TAR 1.00) |
| 0.2 | none (min FAR 7.1e-04) | none (min FAR 1.6e-04) | k=9 (TAR 0.76) |
| 0.5 | none (min FAR 3.4e-03) | none (min FAR 2.0e-03) | none (min FAR 1.0e-03) |

## p_fp = 0.05, FAR < 0.001

| ρ | n = 3 | n = 5 | n = 10 |
|---|---|---|---|
| 0.0 | k=3 (TAR 0.73) | k=4 (TAR 0.92) | k=5 (TAR 1.00) |
| 0.2 | none (min FAR 4.4e-03) | none (min FAR 1.1e-03) | k=9 (TAR 0.76) |
| 0.5 | none (min FAR 1.8e-02) | none (min FAR 1.1e-02) | none (min FAR 5.7e-03) |

## p_fp = 0.05, FAR < 0.0001

| ρ | n = 3 | n = 5 | n = 10 |
|---|---|---|---|
| 0.0 | none (min FAR 1.3e-04) | k=4 (TAR 0.92) | k=5 (TAR 1.00) |
| 0.2 | none (min FAR 4.4e-03) | none (min FAR 1.1e-03) | none (min FAR 1.2e-04) |
| 0.5 | none (min FAR 1.8e-02) | none (min FAR 1.1e-02) | none (min FAR 5.7e-03) |
