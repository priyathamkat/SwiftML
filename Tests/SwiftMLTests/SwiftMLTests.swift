import XCTest
@testable import SwiftML

final class SwiftMLTests: XCTestCase {
    func testTensorInit() throws {
        let zeros = Tensor(withZerosOfShape: [2, 2])
        let ones = Tensor(withOnesOfShape: [2, 2])
        
        XCTAssertEqual(zeros.shape, TensorShape([2, 2]))
        XCTAssertEqual(ones.shape, TensorShape([2, 2]))
    }
    
    func testScalarTensor() throws {
        let zeros = Tensor(withZerosOfShape: [2, 2])
        let ones = Tensor(withOnesOfShape: [2, 2])
        
        XCTAssertEqual((zeros + 1), ones)
        XCTAssertEqual((zeros - 1), -ones)
        XCTAssertEqual((1 - zeros), ones)
        XCTAssertEqual(2 * ones, zeros + 2)
        XCTAssertEqual(ones / 2, zeros + 0.5)
    }
    
    func testTensorTensor() throws {
        let zeros = Tensor(withZerosOfShape: [2, 2])
        let ones = Tensor(withOnesOfShape: [2, 2])
        
        XCTAssertEqual(ones + zeros, ones)
        XCTAssertEqual(ones, Tensor(withOnesOfShape: [2, 2]))
        XCTAssertEqual(zeros, Tensor(withZerosOfShape: [2, 2]))
        XCTAssertEqual(zeros - ones, -ones)
        XCTAssertEqual((2 * ones) * (2 * ones), 4 * ones)
        XCTAssertEqual((2 * ones) / (2 * ones), ones)
    }
    
    func testUniformPerformance() throws {
        let metrics: [XCTMetric] = [XCTClockMetric()]
        let measureOptions = XCTMeasureOptions.default
        measureOptions.iterationCount = 10
        
        measure(metrics: metrics, options: measureOptions) {
            let a = Float.random(in: 0..<1)
            let b = Float.random(in: 0..<1)
            let x = Tensor(withUniformOfShape: [10000, 1000])
            let _ = a * x + b
        }
    }
    
    func testGaussianPerformance() throws {
        let metrics: [XCTMetric] = [XCTClockMetric()]
        let measureOptions = XCTMeasureOptions.default
        measureOptions.iterationCount = 10
        
        measure(metrics: metrics, options: measureOptions) {
            let a = Float.random(in: 0..<1)
            let b = Float.random(in: 0..<1)
            let x = Tensor(withGaussianOfShape: [10000, 1000])
            let _ = a * x + b
        }
    }
    
    func testTensorAdditionPerformance() throws {
        let metrics: [XCTMetric] = [XCTClockMetric()]
        let measureOptions = XCTMeasureOptions.default
        measureOptions.iterationCount = 10
        
        measure(metrics: metrics, options: measureOptions) {
            let x = Tensor(withGaussianOfShape: [1000, 1000])
            let y = Tensor(withGaussianOfShape: [1000, 1000])
            let _ = x + y
        }
    }
}
