"use client"
import React, { useState, useRef } from 'react';
import { Camera, Upload, Loader2, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

const FoodDetectionApp = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, uploading, processing, complete, error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isCameraActive, setIsCameraActive] = useState(false);

  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setStatus('idle');
      setResult(null);
      setError(null);
    }
  };

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }
    } catch (err) {
      setError('Camera access denied or not available');
    }
  };

  // Capture photo from camera
  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (video && canvas) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      canvas.toBlob((blob) => {
        const file = new File([blob], 'camera-capture.png', { type: 'image/png' });
        setSelectedImage(file);
        setPreviewUrl(URL.createObjectURL(blob));
        stopCamera();
        setStatus('idle');
        setResult(null);
      }, 'image/png');
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      setIsCameraActive(false);
    }
  };

  // Upload and analyze image
  const analyzeImage = async () => {
    if (!selectedImage) return;

    setStatus('uploading');
    setError(null);

    try {
      // Step 1: Upload image
      const formData = new FormData();
      formData.append('image', selectedImage);

      const uploadResponse = await fetch(`/api/food`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) throw new Error('Upload failed');

      const { id } = await uploadResponse.json();
      
      // Step 2: Wait for analysis result (long polling)
      setStatus('processing');
      
      const resultResponse = await fetch(`/api/food/${id}/analysis_result`);
      
      if (!resultResponse.ok) throw new Error('Analysis failed');

      const analysisResult = await resultResponse.json();
      
      setResult({
        foodType: analysisResult.food_type,
        id: id,
        // Mock data for demonstration - you'll replace this with real API data
        confidence: analysisResult.confidence * 100,
        volume: Math.random() * 200 + 100,
        calories: Math.floor(Math.random() * 400 + 200),
        protein: Math.floor(Math.random() * 30 + 10),
        recommendation: Math.random() > 0.3 ? 'YES' : 'NOT SURE'
      });
      
      setStatus('complete');
    } catch (err) {
      setError(err.message || 'An error occurred');
      setStatus('error');
    }
  };

  // Reset everything
  const reset = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setStatus('idle');
    setResult(null);
    setError(null);
    stopCamera();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 pt-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🍽️ Food Detection System
          </h1>
          <p className="text-gray-600">
            Upload or capture food images for instant nutritional analysis
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-xl p-6 mb-6">
          {/* Camera View */}
          {isCameraActive && (
            <div className="mb-6">
              <div className="relative bg-black rounded-lg overflow-hidden">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline
                  className="w-full"
                />
              </div>
              <div className="flex gap-4 mt-4">
                <button
                  onClick={capturePhoto}
                  className="flex-1 bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors"
                >
                  📸 Capture Photo
                </button>
                <button
                  onClick={stopCamera}
                  className="px-6 py-3 bg-gray-200 rounded-lg font-semibold hover:bg-gray-300 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Upload/Camera Controls */}
          {!isCameraActive && !previewUrl && (
            <div className="space-y-4">
              <div className="border-4 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-green-400 transition-colors">
                <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                <p className="text-lg text-gray-600 mb-4">
                  Drop an image here or click to upload
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
                >
                  Choose Image
                </button>
              </div>

              <div className="text-center">
                <p className="text-gray-500 mb-3">or</p>
                <button
                  onClick={startCamera}
                  className="bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors inline-flex items-center gap-2"
                >
                  <Camera className="w-5 h-5" />
                  Use Camera
                </button>
              </div>
            </div>
          )}

          {/* Image Preview */}
          {previewUrl && !isCameraActive && (
            <div className="space-y-4">
              <div className="relative">
                <img 
                  src={previewUrl} 
                  alt="Preview" 
                  className="w-full rounded-lg shadow-lg max-h-96 object-contain bg-gray-100"
                />
                {status === 'processing' && (
                  <div className="absolute inset-0 bg-black bg-opacity-50 rounded-lg flex items-center justify-center">
                    <div className="text-center text-white">
                      <Loader2 className="w-12 h-12 animate-spin mx-auto mb-2" />
                      <p className="text-lg font-semibold">Analyzing food...</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              {status === 'idle' && (
                <div className="flex gap-4">
                  <button
                    onClick={analyzeImage}
                    className="flex-1 bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors"
                  >
                    🔍 Analyze Food
                  </button>
                  <button
                    onClick={reset}
                    className="px-6 py-3 bg-gray-200 rounded-lg font-semibold hover:bg-gray-300 transition-colors"
                  >
                    Clear
                  </button>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
                  <XCircle className="w-6 h-6 text-red-500 flex-shrink-0" />
                  <div>
                    <p className="font-semibold text-red-800">Error</p>
                    <p className="text-red-600">{error}</p>
                  </div>
                </div>
              )}

              {/* Results */}
              {status === 'complete' && result && (
                <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-2xl font-bold text-gray-800">
                      Detection Results
                    </h2>
                    <CheckCircle className="w-8 h-8 text-green-600" />
                  </div>

                  {/* Food Type */}
                  <div className="bg-white rounded-lg p-4 shadow">
                    <p className="text-sm text-gray-600 mb-1">Detected Food</p>
                    <p className="text-3xl font-bold text-green-700 capitalize">
                      {result.foodType}
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                      Confidence: {result.confidence.toFixed(1)}%
                    </p>
                  </div>

                  {/* Nutritional Info Grid */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white rounded-lg p-4 shadow">
                      <p className="text-sm text-gray-600">Estimated Volume</p>
                      <p className="text-2xl font-bold text-blue-700">
                        {result.volume.toFixed(0)} ml
                      </p>
                    </div>
                    <div className="bg-white rounded-lg p-4 shadow">
                      <p className="text-sm text-gray-600">Calories</p>
                      <p className="text-2xl font-bold text-orange-600">
                        {result.calories} kcal
                      </p>
                    </div>
                    <div className="bg-white rounded-lg p-4 shadow">
                      <p className="text-sm text-gray-600">Protein</p>
                      <p className="text-2xl font-bold text-purple-600">
                        {result.protein}g
                      </p>
                    </div>
                    <div className="bg-white rounded-lg p-4 shadow">
                      <p className="text-sm text-gray-600">Recommendation</p>
                      <div className="flex items-center gap-2 mt-1">
                        {result.recommendation === 'YES' ? (
                          <>
                            <CheckCircle className="w-6 h-6 text-green-600" />
                            <p className="text-xl font-bold text-green-700">YES</p>
                          </>
                        ) : result.recommendation === 'NO' ? (
                          <>
                            <XCircle className="w-6 h-6 text-red-600" />
                            <p className="text-xl font-bold text-red-700">NO</p>
                          </>
                        ) : (
                          <>
                            <AlertCircle className="w-6 h-6 text-yellow-600" />
                            <p className="text-xl font-bold text-yellow-700">NOT SURE</p>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-4 pt-2">
                    <button
                      onClick={reset}
                      className="flex-1 bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
                    >
                      Analyze Another Image
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Info Card */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">
            How it works
          </h3>
          <ol className="space-y-2 text-gray-600">
            <li>1. Upload an image or capture using your camera</li>
            <li>2. Our AI model detects and segments the food items</li>
            <li>3. Volume and nutritional values are estimated</li>
            <li>4. Get instant recommendations and insights</li>
          </ol>
        </div>
      </div>

      {/* Hidden canvas for camera capture */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};

export default FoodDetectionApp;