public class Tensor<T: Numeric> {
    public let shape: [UInt64]
    public let size: UInt64
    
    private var data: [T]
    
    init(ofShape shape: [UInt64]) {
        self.shape = shape
        let size = shape.reduce(1, { x, y in
            x * y
        })
        self.size = size
        self.data = Array(repeating: 0, count: Int(truncatingIfNeeded: size))
    }
}
