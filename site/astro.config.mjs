import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";

const SITE_URL = "https://sachncs.github.io/kernos";

export default defineConfig({
  site: SITE_URL,
  base: "/kernos",
  output: "static",
  vite: {
    plugins: [tailwindcss()],
  },
  build: {
    inlineStylesheets: "auto",
  },
});
