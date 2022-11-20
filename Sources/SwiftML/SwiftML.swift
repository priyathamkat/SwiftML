public class Tensor<T: Numeric> {
    public let shape: [Int]
    public let size: Int
    
    private var data: [T]
    
    init(ofShape shape: [Int]) {
        self.shape = shape
        let size = shape.reduce(1, { x, y in
            x * y
        })
        self.size = size
        self.data = Array<T>(repeating: 0, count: size)
    }
}
