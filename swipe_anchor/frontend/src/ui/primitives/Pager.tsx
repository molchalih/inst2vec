interface PagerProps {
  count: number;
  index: number;
}

/** Three-dot pager; the active card reads as a wide accent bar. */
export function Pager({ count, index }: PagerProps) {
  return (
    <div className="flex items-center justify-center gap-2">
      {Array.from({ length: count }, (_, i) => (
        <span
          key={i}
          className={[
            "h-1.5 rounded-full transition-all duration-medium ease-out",
            i === index ? "w-6 bg-accent" : "w-1.5 bg-white/20",
          ].join(" ")}
        />
      ))}
    </div>
  );
}
