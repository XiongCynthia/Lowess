## Locally Weighted Scatterplot Smoothing (LOWESS) Implementation

[Lowess](https://raw.githubusercontent.com/XiongCynthia/Lowess/main/Lowess.py) is a class for fitting and predicting data on a LOWESS model.

### Usage

```python
from Lowess import Lowess
lowess = Lowess()
lowess.fit(x_train, y_train)
y_pred = lowess.predict(x_test)
```

More example usages are included in [Lowess_examples.ipynb](https://github.com/XiongCynthia/Lowess/blob/main/Lowess_examples.ipynb).
