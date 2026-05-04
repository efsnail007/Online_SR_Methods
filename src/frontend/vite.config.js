import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootEnvDir = resolve(__dirname, "../..");

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, rootEnvDir, "");

  return {
    envDir: rootEnvDir,
    plugins: [react()],
    server: {
      host: env.FRONTEND_HOST ?? true,
      port: Number(env.FRONTEND_PORT ?? 5173),
    },
  };
});
