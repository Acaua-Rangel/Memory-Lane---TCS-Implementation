// Type declarations for non-JS assets imported via require().
// Metro bundles these files and returns a numeric module ID at runtime.
// The exact runtime value is handled by onnxruntime-react-native / expo-asset.

declare module '*.onnx' {
  const asset: number;
  export default asset;
}

declare module '*.tflite' {
  const asset: number;
  export default asset;
}
