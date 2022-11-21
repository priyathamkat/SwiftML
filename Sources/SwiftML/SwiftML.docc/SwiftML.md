# ``SwiftML``

A simple package for machine learning written entirely in Swift.

## Overview

At the heart of SwiftML is the multi-dimensional Tensor.

```swift
let x = Tensor(withGaussianOfShape: [10, 10])
```

All SwiftML operations are optimized using BLAS/LAPACK/vDSP.
