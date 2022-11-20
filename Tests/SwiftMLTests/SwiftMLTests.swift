import XCTest
@testable import SwiftML

final class SwiftMLTests: XCTestCase {
    func testExample() throws {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(Tensor<Int>(ofShape: [2, 2]).shape, [2, 2])
    }
}
