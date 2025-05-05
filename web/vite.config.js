// vite.config.js
import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  server: {
    port: 3000,
    fs: {
      allow: [
        '.',                          // 기본 프로젝트 루트
        resolve(__dirname, './pkg')  // 외부 디렉토리 명시적으로 허용
      ]
    },
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    }
  },
  assetsInclude: ['**/*.wasm'],
});
