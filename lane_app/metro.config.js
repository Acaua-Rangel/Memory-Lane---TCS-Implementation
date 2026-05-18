const path = require('path');
const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// ── Path aliases (espelha o tsconfig.json paths) ─────────────────────────────
// @/   → src/      (módulos TypeScript)
// @assets/ → assets/  (assets estáticos: imagens, modelos ONNX, etc.)
config.resolver.extraNodeModules = {
  '@': path.join(__dirname, 'src'),
  '@assets': path.join(__dirname, 'assets'),
};

// ── Extensões de modelos ML ───────────────────────────────────────────────────
// Registra como assets (Metro devolve um ID numérico), nunca como source files.
const MODEL_EXTENSIONS = ['onnx', 'tflite', 'bin'];

for (const ext of MODEL_EXTENSIONS) {
  if (!config.resolver.assetExts.includes(ext)) {
    config.resolver.assetExts.push(ext);
  }
  config.resolver.sourceExts = config.resolver.sourceExts.filter((e) => e !== ext);
}

module.exports = config;
