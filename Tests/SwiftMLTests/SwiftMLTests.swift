import XCTest
@testable import SwiftML

final class SwiftMLTests: XCTestCase {
    func testExample() throws {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        let zeros = Tensor(withZerosOfShape: [2, 2])
        let ones = Tensor(withOnesOfShape: [2, 2])
        
        XCTAssertEqual(zeros.shape, TensorShape([2, 2]))
        XCTAssertEqual((zeros + 1), ones)
        XCTAssertEqual((zeros - 1), -ones)
        XCTAssertEqual((1 - zeros), ones)
        XCTAssertEqual(2 * ones, zeros + 2)
    }
}
