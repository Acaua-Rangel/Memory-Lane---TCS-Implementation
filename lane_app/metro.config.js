// Bundler customization for the Expo / React Native app.
// We register `.onnx` so the TCS compression model (and MobileFaceNet) get
// bundled by Metro and become resolvable via `require('./assets/models/x.onnx')`
// + `expo-asset`.
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

if (!config.resolver.assetExts.includes('onnx')) {
  config.resolver.assetExts.push('onnx');
}

module.exports = config;
