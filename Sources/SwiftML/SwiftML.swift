import Accelerate

public class TensorShape {
    public let shape: [Int]
    public var size: Int {
        self.shape.reduce(1, { d_i, d_j in
            d_i * d_j
        })
    }
    public var ndim: Int {
        self.shape.count
    }
    
    public init(_ shape: [Int]) {
        self.shape = shape
    }
}

extension TensorShape: Equatable {
    public static func == (lhs: TensorShape, rhs: TensorShape) -> Bool {
        return lhs.shape == rhs.shape
    }
}

public class Tensor {
    public let shape: TensorShape
    public var size: Int {
        self.shape.size
    }
    public var ndim: Int {
        self.shape.ndim
    }
    
    public var data: [Float]
    
    private init(ofShape shape: TensorShape, withData data: [Float]) {
        assert(shape.size == data.count, "Size of data doesn't match the size inferred from shape")
        self.shape = shape
        self.data = data
    }
    
    convenience init(withZerosOfShape shape: [Int]) {
        let shape = TensorShape(shape)
        let data = Array<Float>(repeating: 0, count: shape.size)
        
        self.init(ofShape: shape, withData: data)
    }
    
    convenience init(withOnesOfShape shape: [Int]) {
        let shape = TensorShape(shape)
        let data = Array<Float>(repeating: 1, count: shape.size)
        
        self.init(ofShape: shape, withData: data)
    }
}

extension Tensor: Equatable {
    public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.shape == rhs.shape && lhs.data == rhs.data
    }
}

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
}
