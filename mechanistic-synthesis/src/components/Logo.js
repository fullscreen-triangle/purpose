import Link from "next/link";
import React from "react";
import { motion } from "framer-motion";

const MotionLink = motion(Link);

export default function Logo() {
  return (
    <div className="flex items-center justify-center">
      <MotionLink
        href="/"
        className="flex items-center justify-center gap-2 rounded-md px-3 py-1.5
                   bg-dark text-light dark:bg-light dark:text-dark
                   font-semibold tracking-tight"
        whileHover={{
          backgroundColor: ["#1b1b1b", "#B63E96", "#1b1b1b"],
          transition: { duration: 1, repeat: Infinity },
        }}
      >
        <span className="text-base">MS</span>
        <span className="text-xs font-normal opacity-70 hidden sm:inline">
          mechanistic-synthesis
        </span>
      </MotionLink>
    </div>
  );
}
