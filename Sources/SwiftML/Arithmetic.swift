import Accelerate

// Scalar-Tensor operations
@available(macOS 10.15, *)
extension Tensor {
    public static func + (left: Tensor, right: Float) -> Tensor {
        let shape = left.shape
        let data: [Float] = vDSP.add(right, left.data)
        return Tensor(ofShape: shape, withData: data)
    }
    
    public static func + (left: Float, right: Tensor) -> Tensor {
        return right + left
    }
    
    public static prefix func - (tensor: Tensor) -> Tensor {
        return Tensor(ofShape: tensor.shape, withData: vDSP.negative(tensor.data))
    }
    
    public static func - (left: Tensor, right: Float) -> Tensor {
        let shape = left.shape
        let data: [Float] = vDSP.add(-right, left.data)
        return Tensor(ofShape: shape, withData: data)
    }
    
    public static func - (left: Float, right: Tensor) -> Tensor {
        return -right + left
    }
    
    public static func * (left: Tensor, right: Float) -> Tensor {
        let shape = left.shape
        let data: [Float] = vDSP.multiply(right, left.data)
        return Tensor(ofShape: shape, withData: data)
    }
    
    public static func * (left: Float, right: Tensor) -> Tensor {
        return right * left
    }
    
    public static func / (left: Tensor, right: Float) -> Tensor {
        let shape = left.shape
        let data: [Float] = vDSP.divide(left.data, right)
        return Tensor(ofShape: shape, withData: data)
    }
    
    public static func / (left: Float, right: Tensor) -> Tensor {
        let shape = right.shape
        let data: [Float] = vDSP.divide(left, right.data)
        return Tensor(ofShape: shape, withData: data)
    }
}

// Tensor-Tensor operations
@available(macOS 10.15, *)
extension Tensor {
    public static func + (left: Tensor, right: Tensor) -> Tensor {
        let shape = left.shape
        precondition(shape == right.shape)
        var data: [Float] = left.data
        cblas_saxpy(Int32(left.size), 1.0, &right.data, 1, &data, 1)
        return Tensor(ofShape: shape, withData: data)
    }
    
    public static func - (left: Tensor, right: Tensor) -> Tensor {
        let shape = left.shape
        precondition(shape == right.shape)
        var data: [Float] = left.data
        cblas_saxpy(Int32(left.size), -1.0, &right.data, 1, &data, 1)
        return Tensor(ofShape: shape, withData: data)
    }
    
    public static func * (left: Tensor, right: Tensor) -> Tensor {
        let shape = left.shape
        precondition(shape == right.shape)
        let data: [Float] = vDSP.multiply(left.data, right.data)
        return Tensor(ofShape: shape, withData: data)
    }
    
    public static func / (left: Tensor, right: Tensor) -> Tensor {
        let shape = left.shape
        precondition(shape == right.shape)
        let data: [Float] = vDSP.divide(left.data, right.data)
        return Tensor(ofShape: shape, withData: data)
    }
}
