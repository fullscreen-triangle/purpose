import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        routeGraph: resolve(__dirname, 'route-graph.html'),
      },
    },
  },
  server: {
    port: 5173,
    open: true,
  },
});
