import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  // Served at the site root by the backend's SPA mount (StaticFiles at "/"),
  // and by the Telegram Mini App at the same origin — so assets must resolve
  // from "/", not a subpath (a "/swipe-anchor/" base 404s every asset).
  base: "/",
  plugins: [react()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: { port: 5174 },
});
