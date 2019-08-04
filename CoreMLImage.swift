//
//  Created by Daryl Rowland on 15/12/2017.
//  Copyright © Jigsaw XYZ
//

import Foundation
import UIKit
import CoreML
import AVFoundation
import Vision

@available(iOS 11.0, *)
@objc(CoreMLImage)
public class CoreMLImage: UIView, AVCaptureVideoDataOutputSampleBufferDelegate {
	
	var bridge: RCTEventDispatcher!
	var captureSession: AVCaptureSession?
	var videoPreviewLayer: AVCaptureVideoPreviewLayer?
	var model: VNCoreMLModel?
	var lastClassification: String = ""
	var onClassification: RCTBubblingEventBlock?
	var lastComputation: Int64 = 0
	
	required public init(coder aDecoder: NSCoder) {
		super.init(coder: aDecoder)!
	}
	
	override init(frame: CGRect) {
		super.init(frame: frame)
		self.frame = frame;
	}
	
	
	func capture(_ captureOutput: AVCaptureFileOutput!, didStartRecordingToOutputFileAt fileURL: URL!, fromConnections connections: [Any]!) {
	}
	
	public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
		let now = Int64(NSDate().timeIntervalSince1970 * 1000)
		if (now > self.lastComputation + 300) {
			let img = self.imageFromSampleBuffer(sampleBuffer: sampleBuffer)
			
			self.lastComputation = now
			runMachineLearning(img: img)
			
		}
		else {
			print("SKIPPING CLASSIFICATION")
		}
		
	}
	
	func runMachineLearning(img: CIImage) {
		if (self.model != nil) {
			let request = VNCoreMLRequest(model: self.model!, completionHandler: { [weak self] request, error in
				self?.processClassifications(for: request, error: error)
			})
			request.imageCropAndScaleOption = .centerCrop
			
			let orientation = CGImagePropertyOrientation.up
			
			DispatchQueue.global(qos: .userInitiated).async {
				let handler = VNImageRequestHandler(ciImage: img, orientation: orientation)
				do {
					try handler.perform([request])
				} catch {
					print("Failed to perform classification.\n\(error.localizedDescription)")
				}
			}
		}
	}
	
	func processClassifications(for request: VNRequest, error: Error?) {
		DispatchQueue.main.async {
			guard let results = request.results else {
				print("Unable to classify image")
				print(error!.localizedDescription)
				return
			}
			
			let classifications = results as! [VNClassificationObservation]
			
			//        print(classifications)
			
			var classificationArray = [Dictionary<String, Any>]()
			//        var orderedArray = classifications.sorted(by: { $0.confidence < $1.confidence })
			
			
			//        print(orderedArray)
			var x = 0
			for classification in classifications {
				classificationArray.append(["identifier": classification.identifier, "confidence": classification.confidence])
				if (x > 500) {
					break
				}
				x += 1
			}
			print(classificationArray)
			
			//        classifications.forEach{classification in
			//          classificationArray.append(["identifier": classification.identifier, "confidence": classification.confidence])
			//
			//        }
			
			self.onClassification!(["classifications": classificationArray])
			
		}
		
	}
	
	func imageFromSampleBuffer(sampleBuffer : CMSampleBuffer) -> CIImage
	{
		let imageBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)!
		
		let ciimage : CIImage = CIImage(cvPixelBuffer: imageBuffer)
		return ciimage
	}
	
	override public func layoutSubviews() {
		super.layoutSubviews()
		let view = UIView(frame: CGRect(x: 0, y: 0, width: self.frame.width,
										height: self.frame.height))
		
		let captureDevice = AVCaptureDevice.default(for: .video)
		
		do {
			if (captureDevice != nil) {
				let input = try AVCaptureDeviceInput(device: captureDevice!)
				self.captureSession = AVCaptureSession()
				self.captureSession?.addInput(input)
				self.captureSession!.sessionPreset = AVCaptureSession.Preset.low;
				videoPreviewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession!)
				videoPreviewLayer?.videoGravity = AVLayerVideoGravity.resizeAspectFill
				videoPreviewLayer?.frame = view.layer.bounds
				
				view.layer.addSublayer(videoPreviewLayer!)
				self.addSubview(view)
				
				let videoDataOutput = AVCaptureVideoDataOutput()
				let queue = DispatchQueue(label: "xyz.jigswaw.ml.queue")
				videoDataOutput.setSampleBufferDelegate(self, queue: queue)
				guard (self.captureSession?.canAddOutput(videoDataOutput))! else {
					fatalError()
				}
				self.captureSession?.addOutput(videoDataOutput)
				self.captureSession?.startRunning()
			}
			
		} catch {
			print(error)
		}
		
	}
	
	@objc(setModelFile:) public func setModelFile(modelPath: String) {
		//    print("Setting model file to: " + modelFile)
		//let path = Bundle.main.url(forResource: modelFile, withExtension: "mlmodelc")
		let path = NSURL.fileURL(withPath: modelPath)
		
		do {
			let modelUrl = try MLModel.compileModel(at: path)
			let model = try MLModel.init(contentsOf: modelUrl)
			self.model = try VNCoreMLModel(for: model)
			
		} catch {
			print("Error")
		}
		
	}
	
	@objc(setOnClassification:) public func setOnClassification(onClassification: @escaping RCTBubblingEventBlock) {
		self.onClassification = onClassification
	}
	
	
	
}

