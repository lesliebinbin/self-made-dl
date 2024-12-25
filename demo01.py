print("hello")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

x = np.arange(0, 100)

x

y = x ** 2

y

plt.plot(x, y)
plt.show()
