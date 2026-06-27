import type { TimeRangeKey } from "../../lib/dashboard/types";
import { TIME_RANGE_OPTIONS } from "../../lib/dashboard/events";

interface RangeControlsProps {
  selectedRange: TimeRangeKey;
  onChange: (range: TimeRangeKey) => void;
}

export function RangeControls({ selectedRange, onChange }: RangeControlsProps) {
  return (
    <div className="flex items-center gap-2 overflow-x-auto pb-1 rounded-full bg-zinc-900/40 border border-zinc-700/40 p-1.5 backdrop-blur-sm">
      {TIME_RANGE_OPTIONS.map((option) => {
        const isActive = selectedRange === option.key;
        return (
          <button
            key={option.key}
            onClick={() => onChange(option.key)}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-all duration-200 whitespace-nowrap ${
              isActive
                ? "bg-gradient-to-r from-burgundy-700/85 to-burgundy-500/80 text-burgundy-50 border-burgundy-500/70 shadow-[0_0_16px_rgba(192,64,85,0.35)]"
                : "bg-zinc-900/50 text-zinc-400 border-zinc-700/60 hover:text-zinc-200 hover:border-zinc-500/70 hover:bg-zinc-800/70"
            }`}
            type="button"
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
