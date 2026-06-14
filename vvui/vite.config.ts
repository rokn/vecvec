import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The vecvec REST gateway listens on 127.0.0.1:6333 (override with VECVEC_REST_ADDR
// on the server). We proxy `/api/*` to it so the browser talks same-origin in dev —
// no CORS round-trips — while the server's permissive CORS layer covers direct use.
const TARGET = process.env.VECVEC_REST_URL ?? "http://127.0.0.1:6333";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5273,
    proxy: {
      "/api": {
        target: TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
