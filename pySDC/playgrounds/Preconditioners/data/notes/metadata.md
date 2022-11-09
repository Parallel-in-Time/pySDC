# Preconditioners 
We supply some information about the preconditioners here
| name | source | parallelizable | normalized | semi-diagonal | random IG |
|------|--------|----------------|------------|---------------|-----------|
| Semi-Diagonal | optimization | True | False | True | False |
| LU | [Martin Weiser](https://doi.org/10.1007/s10543-014-0540-y) | False | False | False | True |
| Diagonal | optimization | False | False | False | False |
| Implicit Euler | [Dutt et al.](https://doi.org/10.1023/A:1022338906936) | False | False | False | True |
| MIN | [Robert](https://doi.org/10.1007/s00791-018-0298-x) | True | False | False | True |
| normalized | optimization | True | True | False | False |
| MIN3 | Anonymous | True | False | False | True |
