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

public class Tensor {
    public let shape: TensorShape
    public var size: Int {
        self.shape.size
    }
    public var ndim: Int {
        self.shape.ndim
    }
    
    private var data: [Float]
    
    private init(ofShape shape: TensorShape, withData data: [Float]) {
        assert(shape.size == data.count, "Size of data doesn't match the size inferred from shape")
        self.shape = shape
        self.data = data
    }
    
    convenience init(withZerosOfShape shape: [Int]) {
        let shape = TensorShape(shape)
        var data = Array<Float>(repeating: 0, count: shape.size)
        
        self.init(ofShape: shape, withData: data)
    }
    
    convenience init(withOnesOfShape shape: [Int]) {
        let shape = TensorShape(shape)
        var data = Array<Float>(repeating: 1, count: shape.size)
        
        self.init(ofShape: shape, withData: data)
    }
}
