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
