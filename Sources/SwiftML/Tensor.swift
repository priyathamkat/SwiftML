import Accelerate

private func generateRandom(d: Int, n: Int) -> [Float] {
    // Based on https://netlib.org/lapack/explore-html/df/dd1/group___o_t_h_e_rauxiliary_ga379b09e3d4e7635db614d3b3973db5e7.html
    // Returns an Array of size `n` filled with random numbers
    // Distribution of the random numbers depends on `d`: `d` = 1 for Uniform (0, 1), `d` = 3 for Normal(0, 1)
    let data: [Float] = Array<Float>(unsafeUninitializedCapacity: n) { buffer, count in
        var d_i32: Int32 = Int32(d)
        var n_i32: Int32 = Int32(n)
        var seed: [Int32] = Array<Int32>(repeating: 0, count: 4)
        for i in 0...2 {
            seed[i] = Int32.random(in: 0...4095)
        }
        seed[3] = 2 * Int32.random(in: 0...2047) + 1
        slarnv_(&d_i32, &seed, &n_i32, buffer.baseAddress)
        count = n
    }
    return data
}

public class Tensor: CustomStringConvertible {
    public let shape: TensorShape
    public var size: Int {
        self.shape.size
    }
    public var ndim: Int {
        self.shape.ndim
    }
    
    public var data: [Float]
    
    internal init(ofShape shape: TensorShape, withData data: [Float]) {
        precondition(shape.size == data.count, "Size of data doesn't match the size inferred from shape")
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
    
    @available(macOS 10.15, *)
    convenience init(withUniformOfShape shape: [Int], low: Float = 0.0, high: Float = 1.0) {
        // Create a tensor filled with random numbers from a uniform distribution between `low` and `high`
        let shape = TensorShape(shape)
        var data: [Float] = generateRandom(d: 1, n: shape.size)
        
        if (low != 0) {
            data = vDSP.add(multiplication: (data, high - low), low)
        } else {
            data = vDSP.multiply(high, data)
        }
        
        self.init(ofShape: shape, withData: data)
    }
    
    @available(macOS 10.15, *)
    convenience init(withGaussianOfShape shape: [Int], mean: Float = 0.0, std: Float = 1.0) {
        // Create a tensor filled with random numbers from a gaussian distribution with mean `mean` and standard deviation `std`
        let shape = TensorShape(shape)
        var data: [Float] = generateRandom(d: 3, n: shape.size)
        
        if (mean != 0 && std != 1) {
            data = vDSP.add(multiplication: (data, std), mean)
        } else if (mean == 0 && std != 1) {
            data = vDSP.multiply(std, data)
        } else if (mean != 0 && std == 1) {
            data = vDSP.add(mean, data)
        }
        
        self.init(ofShape: shape, withData: data)
    }
    
    public var description: String {
        return String(describing: self.data)
    }
}

extension Tensor: Equatable {
    public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.shape == rhs.shape && lhs.data == rhs.data
    }
}
