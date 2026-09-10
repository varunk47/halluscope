import gsap from "gsap";
import { useGSAP } from "@gsap/react";

gsap.registerPlugin(useGSAP);

/** People who asked for less motion get instant states, not slower tweens. */
export function reducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Roll a number readout from its current value to `to`. Writes straight to the
 * DOM node so React does not re-render sixty times a second for one figure.
 */
export function rollNumber(
  el: HTMLElement | null,
  to: number,
  format: (v: number) => string,
  duration = 0.9,
) {
  if (!el) return;
  const state = { v: Number(el.dataset.value ?? 0) };
  if (reducedMotion()) {
    el.textContent = format(to);
    el.dataset.value = String(to);
    return;
  }
  gsap.to(state, {
    v: to,
    duration,
    ease: "power3.out",
    overwrite: true,
    onUpdate: () => {
      el.textContent = format(state.v);
    },
    onComplete: () => {
      el.dataset.value = String(to);
    },
  });
}

export { gsap, useGSAP };
