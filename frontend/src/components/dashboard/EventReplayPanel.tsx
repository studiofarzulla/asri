import { Pause, Play, RotateCcw } from "lucide-react";
import type { CrisisEvent } from "../../lib/dashboard/events";
import type { TimeseriesPoint } from "../../lib/dashboard/types";

interface EventReplayPanelProps {
  events: CrisisEvent[];
  selectedEventId: string;
  onSelectEvent: (eventId: string) => void;
  focusReplay: boolean;
  onToggleFocus: () => void;
  replayIndex: number;
  replayMaxIndex: number;
  onSeek: (nextIndex: number) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onReset: () => void;
  currentReplayPoint: TimeseriesPoint | null;
}

export function EventReplayPanel({
  events,
  selectedEventId,
  onSelectEvent,
  focusReplay,
  onToggleFocus,
  replayIndex,
  replayMaxIndex,
  onSeek,
  isPlaying,
  onTogglePlay,
  onReset,
  currentReplayPoint,
}: EventReplayPanelProps) {
  return (
    <section className="asri-glass p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-zinc-100 font-mono tracking-tight">Event Replay</h2>
          <p className="text-xs text-zinc-400">
            Scrub through crisis windows to inspect ASRI signal buildup
          </p>
        </div>
        <button
          type="button"
          onClick={onToggleFocus}
          className={`px-3 py-1.5 rounded-lg text-xs border transition-all ${
            focusReplay
              ? "bg-gradient-to-r from-burgundy-700/80 to-burgundy-500/75 border-burgundy-500/70 text-burgundy-50 shadow-[0_0_16px_rgba(192,64,85,0.35)]"
              : "bg-zinc-900/50 border-zinc-700/60 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500/70"
          }`}
        >
          {focusReplay ? "Replay Focus On" : "Replay Focus Off"}
        </button>
      </div>

      <div className="grid md:grid-cols-[220px,1fr] gap-4">
        <select
          value={selectedEventId}
          onChange={(event) => onSelectEvent(event.target.value)}
          className="bg-zinc-900/70 border border-zinc-700/60 rounded-lg px-3 py-2 text-sm text-zinc-200 outline-none focus:border-burgundy-500/70"
        >
          {events.map((item) => (
            <option key={item.id} value={item.id}>
              {item.name}
            </option>
          ))}
        </select>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onTogglePlay}
            disabled={!focusReplay || replayMaxIndex <= 0}
            className="p-2 rounded-lg border border-zinc-700/60 bg-zinc-900/70 text-zinc-200 disabled:opacity-40 hover:border-zinc-500/70 transition-colors"
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </button>
          <button
            type="button"
            onClick={onReset}
            disabled={!focusReplay || replayMaxIndex <= 0}
            className="p-2 rounded-lg border border-zinc-700/60 bg-zinc-900/70 text-zinc-200 disabled:opacity-40 hover:border-zinc-500/70 transition-colors"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(replayMaxIndex, 0)}
            value={Math.min(replayIndex, replayMaxIndex)}
            onChange={(event) => onSeek(Number(event.target.value))}
            className="w-full accent-burgundy-500 cursor-pointer disabled:cursor-not-allowed"
            disabled={!focusReplay || replayMaxIndex <= 0}
          />
        </div>
      </div>

      <div className="text-xs text-zinc-300 rounded-lg border border-zinc-700/40 bg-zinc-900/50 px-3 py-2">
        {currentReplayPoint
          ? `Replay point: ${currentReplayPoint.date} · ASRI ${currentReplayPoint.asri.toFixed(2)}`
          : "Replay data unavailable for selected event range."}
      </div>
    </section>
  );
}
