import type { Config } from "tailwindcss";

// Loosely matches mechanistic-synthesis/tailwind.config.js's palette
// (primary/primaryDark, dark-mode class strategy) for visual family
// resemblance, without being coupled to that app.
const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        dark: "#1b1b1b",
        light: "#f5f5f5",
        primary: "#B63E96",
        primaryDark: "#58E6D9",
      },
    },
  },
  plugins: [],
};

export default config;
