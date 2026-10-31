/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` proxies the API to a running det-depth-distance-web
// (override with DDD_API, e.g. DDD_API=http://127.0.0.1:8765).
const api = process.env.DDD_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": api } },
  build: { outDir: "dist", emptyOutDir: true, sourcemap: true },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
