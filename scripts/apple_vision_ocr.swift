#!/usr/bin/env swift

import Foundation
import ImageIO
import Vision

func emit(_ payload: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

func recognize(_ argument: String) {
    autoreleasepool {
        let url = URL(fileURLWithPath: argument)
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            emit(["path": argument, "error": "could not decode image"])
            return
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.recognitionLanguages = ["en-US"]
        request.minimumTextHeight = 0.008

        do {
            try VNImageRequestHandler(cgImage: image).perform([request])
            let lines = (request.results ?? []).compactMap {
                $0.topCandidates(1).first?.string
            }
            emit(["path": argument, "lines": lines])
        } catch {
            emit(["path": argument, "error": error.localizedDescription])
        }
    }
}

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments == ["--stdin"] {
    var completed = 0
    while let path = readLine() {
        recognize(path)
        completed += 1
        if completed % 1_000 == 0 {
            FileHandle.standardError.write(Data("OCR processed \(completed) images\n".utf8))
        }
    }
} else {
    for argument in arguments {
        recognize(argument)
    }
}
