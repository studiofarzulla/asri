import { FlaskConical, ShieldCheck } from "lucide-react";
import type { ValidationResponse } from "../../lib/dashboard/types";

interface ValidationSnapshotProps {
  validation: ValidationResponse | null;
  loading: boolean;
  error: string | null;
}

export function ValidationSnapshot({
  validation,
  loading,
  error,
}: ValidationSnapshotProps) {
  return (
    <section className="bg-zinc-900/35 backdrop-blur-sm rounded-2xl border border-zinc-700/40 p-5 shadow-[0_18px_38px_rgba(0,0,0,0.28)]">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-burgundy-300" />
          <h2 className="text-sm font-semibold text-zinc-100">Validation Snapshot</h2>
        </div>
        <ShieldCheck className="h-4 w-4 text-emerald-300" />
      </div>

      {loading && (
        <div className="text-xs text-zinc-400">Loading validation telemetry...</div>
      )}

      {!loading && error && (
        <div className="text-xs text-amber-300 bg-amber-950/30 border border-amber-900/40 rounded-lg px-3 py-2">
          Validation endpoint unavailable: {error}
        </div>
      )}

      {!loading && !error && validation && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="rounded-lg border border-zinc-700/50 bg-gradient-to-b from-zinc-800/55 to-zinc-900/55 px-3 py-2">
            <p className="text-[11px] text-zinc-400">Methodology Profile</p>
            <p className="text-sm font-medium text-zinc-200">
              {validation.event_study.methodology_profile ?? "paper_v2"}
            </p>
          </div>
          <div className="rounded-lg border border-zinc-700/50 bg-gradient-to-b from-zinc-800/55 to-zinc-900/55 px-3 py-2">
            <p className="text-[11px] text-zinc-400">Detection Rate</p>
            <p className="text-sm font-medium text-zinc-200">
              {validation.event_study.summary?.detection_rate !== undefined
                ? `${(validation.event_study.summary.detection_rate * 100).toFixed(0)}%`
                : "n/a"}
            </p>
          </div>
          <div className="rounded-lg border border-zinc-700/50 bg-gradient-to-b from-zinc-800/55 to-zinc-900/55 px-3 py-2">
            <p className="text-[11px] text-zinc-400">Avg Lead</p>
            <p className="text-sm font-medium text-zinc-200">
              {validation.event_study.summary?.avg_lead_time !== undefined
                ? `${validation.event_study.summary.avg_lead_time.toFixed(1)}d`
                : "n/a"}
            </p>
          </div>
          <div className="rounded-lg border border-zinc-700/50 bg-gradient-to-b from-zinc-800/55 to-zinc-900/55 px-3 py-2">
            <p className="text-[11px] text-zinc-400">Avg CAS</p>
            <p className="text-sm font-medium text-zinc-200">
              {validation.event_study.summary?.avg_cas !== undefined
                ? validation.event_study.summary.avg_cas.toFixed(1)
                : "n/a"}
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
