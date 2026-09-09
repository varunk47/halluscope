import type { ReactNode } from "react";
import type { Label, ReviewStatus, Role } from "../api";

export function SectionLabel({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 mb-3">
      <span className="label">{children}</span>
      {right && <span className="text-[11px] text-muted">{right}</span>}
    </div>
  );
}

export function Panel({
  children,
  className = "",
  padded = true,
}: {
  children: ReactNode;
  className?: string;
  padded?: boolean;
}) {
  return <section className={`panel ${padded ? "p-4 md:p-5" : ""} ${className}`}>{children}</section>;
}

export function PageHeader({
  title,
  subtitle,
  right,
}: {
  title: string;
  subtitle?: ReactNode;
  right?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
      <div>
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="mt-1 text-[13px] text-muted max-w-2xl leading-5">{subtitle}</p>}
      </div>
      {right && <div className="flex items-center gap-2">{right}</div>}
    </div>
  );
}

export const LABEL_COLOR: Record<Label, string> = {
  specified: "#22d3ee",
  underspecified: "#f97316",
  inconsistent: "#a78bfa",
};

export function LabelChip({ label }: { label: Label | string }) {
  const cls =
    label === "specified"
      ? "border-safe/40 text-safe bg-safe/10"
      : label === "underspecified"
        ? "border-risk/40 text-risk bg-risk/10"
        : label === "inconsistent"
          ? "border-incons/40 text-incons bg-incons/10"
          : "border-line text-muted";
  return <span className={`chip ${cls}`}>{label}</span>;
}

const STATUS_CLS: Record<ReviewStatus, string> = {
  pending: "border-line-2 text-muted",
  approved: "border-safe/40 text-safe bg-safe/10",
  edited: "border-incons/40 text-incons bg-incons/10",
  rejected: "border-risk-hot/40 text-risk-hot bg-risk-hot/10",
};

export function StatusChip({ status }: { status: ReviewStatus }) {
  return <span className={`chip ${STATUS_CLS[status] ?? "border-line text-muted"}`}>{status}</span>;
}

export function RoleTag({ role }: { role: Role }) {
  return (
    <span
      className={`label !tracking-[0.12em] ${role === "user" ? "text-safe" : "text-incons"}`}
    >
      {role}
    </span>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="panel grid-texture p-8 text-center">
      <div className="text-[15px] font-medium text-text">{title}</div>
      {children && <div className="mt-2 text-[13px] text-muted leading-5 max-w-lg mx-auto">{children}</div>}
    </div>
  );
}

export function Spinner({ label = "loading" }: { label?: string }) {
  return (
    <div className="inline-flex items-center gap-2 text-[12px] text-muted">
      <span className="h-1.5 w-1.5 rounded-full bg-safe animate-pulseDot" />
      {label}
    </div>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "default" | "safe" | "risk" | "incons";
}) {
  const color =
    tone === "safe"
      ? "text-safe"
      : tone === "risk"
        ? "text-risk"
        : tone === "incons"
          ? "text-incons"
          : "text-text";
  return (
    <div className="min-w-0">
      <div className="label">{label}</div>
      <div className={`mono mt-1 text-[18px] leading-6 ${color}`}>{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-dim">{hint}</div>}
    </div>
  );
}
