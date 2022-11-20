import XCTest
@testable import SwiftML

final class SwiftMLTests: XCTestCase {
    func testExample() throws {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        let x = Tensor(withZerosOfShape: [2, 2])
        XCTAssertEqual(x.shape, TensorShape([2, 2]))
        XCTAssertEqual((x + 1).data, [1, 1, 1, 1])
        XCTAssertEqual((x - 1).data, [-1, -1, -1, -1])
        XCTAssertEqual((1 - x).data, [1, 1, 1, 1])
    }
}
