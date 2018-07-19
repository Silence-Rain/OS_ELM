## OS_ELM

Python3 implementation of Online Sequential Extreme Learning Machine

#### Requirements

- Python ^3.6.3
- numpy ^1.14.2

#### Features

- Provides normalization in the model (**Only 0-1 normalization provided currently**). 
- Automatically calculates the range of labels according to the given data. No extra parameter needed for normalization.
- Calculates the training error rate.

#### APIs

- **fit_init()**
  Initial training.

  | parameters/return value | type     | description                           |
  | ----------------------- | -------- | ------------------------------------- |
  | data                    | np.array | train set                             |
  | label_index             | int      | the index of label column (default 0) |
  | return                  | OS_ELM   | trained network                       |

- **fit_train()**

  Update the network using online data.

  | parameters/return value | type     | description                               |
  | ----------------------- | -------- | ----------------------------------------- |
  | data                    | np.array | online train set (**1 record each time**) |
  | label_index             | int      | the index of label column (default 0)     |
  | return                  | OS_ELM   | updated network                           |

- **predict()**

  Predict labels using current network.

  | parameters/return value | type     | description      |
  | ----------------------- | -------- | ---------------- |
  | data                    | np.array | test set         |
  | return                  | list     | predicted labels |

  